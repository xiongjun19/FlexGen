#coding=utf8

import argparse
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from accelerate import (infer_auto_device_map, init_empty_weights,
    load_checkpoint_and_dispatch)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import OPTForCausalLM
import torch

from flexgen.timer import timers
from flexgen.utils import (GB, project_decode_latency,
    write_benchmark_log)
from flexgen.opt_config import (get_opt_config,
    disable_torch_init, disable_hf_opt_init)

class Infer(object):
    def __init__(self, args):
            '''
            args.model, args.batch_size, args.prompt_len, args.gen_len,
            args.cut_gen_len, args.cpu_offload, args.disk_offload,
            os.path.abspath(os.path.expanduser(args.offload_dir)),
            args.int8, num_nodes, num_gpus_per_node, use_deepspeed,
            args.dummy, args.log_file, args.pkl_file,
            args.no_log, args.verbose
            '''
        self.model_name = args.model
        self.local_rank = args.local_rank
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name.replace("175b", "66b"), padding_side="left")
        self.offload_dir = os.path.abspath(os.path.expanduser(args.offload_dir))
        dtype=torch.float16
        if args.int8:
            dtype=torch.int8
        dummy_weights = None
        self.model = get_ds_opt_model(model_name, dtype, args.cpu_offload, args.disk_offload,
            self.offload_dir, dummy_weights)

        prompts = ["Paris is the capital city of"]
        input_ids = self.tokenizer(
                prompts, return_tensors="pt").input_ids.cuda()

        generate_kwargs_warmup = dict(max_new_tokens=1, do_sample=False)
        with torch.no_grad():
            self.model.generate(input_ids=input_ids, **generate_kwargs_warmup)

    def run(self, prompts, gen_len=64):
        input_ids =  self.tokenizer(prompts, return_tensor='pt').input_ids.cuda()
        generate_kwargs = dict(max_new_tokens=gen_len, do_sample=False)
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
            if args.local_rank > 0:
                return output_ids
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return outputs




def meta_to_cpu(container, dtype=None):
    if isinstance(container, torch.Tensor):
        return torch.empty(*container.shape, dtype=dtype or container.dtype)
    elif isinstance(container, tuple):
        return tuple(meta_to_cpu(x, dtype) for x in container)
    elif isinstance(container, dict):
        return dict((k, meta_to_cpu(v, dtype)) for k, v in container.items())
    else:
        raise ValueError(f"Invalid type: {container}")


def realize_meta_module(module, dtype=None, device=None):
    for name, child in module.named_children():
        realize_meta_module(child, dtype, device)

    keys = list(module._parameters.keys())
    for k in keys:
        v = module._parameters[k]
        if v is not None:
            module._parameters[k] = torch.nn.Parameter(
                torch.empty(*v.shape, dtype=dtype or v.dtype,
                    device=device or v.device))

    keys = list(module._buffers.keys())
    for k in keys:
        v = module._buffers[k]
        assert v is None


def get_model_config(model_name):
    if "175b" in model_name:
        config = AutoConfig.from_pretrained("facebook/opt-66b")
        config.hidden_size = 12288
        config.word_embed_proj_dim = 12288
        config.ffn_dim = 12288 * 4
        config.num_attention_heads = 96
        config.num_hidden_layers = 96
    else:
        config = AutoConfig.from_pretrained(model_name)

    return config


def get_ds_opt_model(model_name, dtype, cpu_offload, disk_offload, offload_dir,
                     dummy_weights):
    import deepspeed
    import torch.distributed as dist
    from transformers.deepspeed import HfDeepSpeedConfig

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = bool(args.pin_memory)

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": hidden_size * hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": args.batch_size,
        "wall_clock_breakdown": False,
        "tensor_parallel": {"tp_size": WORLD_SIZE},
    }

    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory)

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=2 * GB,
        )
        ds_config["aio"] = {
          "block_size": 1048576,
          "queue_depth": 8,
          "thread_count": 1,
          "single_submit": False,
          "overlap_events": True,
        }

    dschf = HfDeepSpeedConfig(ds_config)

    model = OPTForCausalLM.from_pretrained(
        dummy_weights or model_name, torch_dtype=dtype)
    model = model.eval()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model


def get_hf_opt_model(model_name, dtype, cpu_offload, disk_offload, offload_dir,
                     num_gpus, dummy_weights):
    if num_gpus == 1 and dtype != torch.int8:
        # Here we use a custom device_map instead of device_map == "auto"
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        if cpu_offload:
            # NOTE: We must put some weights on GPU. Otherwise, huggingface reports errors.
            device_map = {
                "model.decoder.embed_tokens.weight": 0,
                "model.decoder.embed_positions.weight": 0,
                "model.decoder.final_layer_norm": "cpu",
                "model.decoder.layers": "cpu",
                "lm_head.weight": 0,
            }
        elif disk_offload:
            device_map = {
                "model.decoder.embed_tokens.weight": 0,
                "model.decoder.embed_positions.weight": 0,
                "model.decoder.final_layer_norm": "disk",
                "model.decoder.layers": "disk",
                "lm_head.weight": 0,
            }
        else:
            device_map = None
        max_memory = None
    else:
        # Here we use device_map == "auto", but set a low `max_memory` threshold
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        device_map = "auto"
        if cpu_offload:
            # `max_memory` should be larger than the embedding.
            # We use 2GB here because the embeding of opt-175b is 1.2GB.
            max_memory = {k: "2GB" for k in range(num_gpus)}
        elif disk_offload:
            max_memory = {k: "2GB" for k in range(num_gpus)}
        else:
            max_memory = {k: "14GB" for k in range(num_gpus)}
        max_memory["cpu"] = "160GB"

    if dtype == torch.int8:
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {"torch_dtype": dtype}

    disable_torch_init()
    model = OPTForCausalLM.from_pretrained(dummy_weights or model_name,
        device_map=device_map, max_memory=max_memory,
        offload_folder=offload_dir, **kwargs)
    if device_map is None:
        model.cuda()

    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dummy", action="store_true",
        help="Use dummy weights for benchmark purposes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--disk-offload", action="store_true")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir")
    parser.add_argument("--int8", action="store_true")

    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--pkl-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args()

