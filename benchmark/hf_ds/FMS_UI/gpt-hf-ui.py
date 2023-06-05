# -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # normal 
import logging
import sys
import gradio as gr
import torch
import gc
from app_modules.utils import *
from app_modules.presets import *
from app_modules.overwrites import *
import mmap, time, _io
from transformers import AutoTokenizer, AutoModelForCausalLM
import pycuda.driver as cuda

import argparse
import multiprocessing as mp
import os
import pickle
import time
import random
import numpy as np

from accelerate import (infer_auto_device_map, init_empty_weights,
    load_checkpoint_and_dispatch)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import OPTForCausalLM
import torch
random.seed(42) 
torch.manual_seed(42)
from flexgen.timer import timers
from flexgen.utils import (GB, project_decode_latency,
    write_benchmark_log)
from flexgen.opt_config import (get_opt_config,
    disable_torch_init, disable_hf_opt_init)







logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

model_name = "facebook_opt-1.3b"
# model_name = "facebook_opt-350m"
model_path = f"models/{model_name}/"

# tokenizer,model,device = load_tokenizer_and_model(base_model,adapter_model)
device = 'cuda'

use_cuda=True

def log_time(st):
    print(f'\033[36mtook {round(time.time()-st,3)}s\033[0m')  
    return time.time()  

def log_message(text, color):
    print(f'{color}{text}\033[0m')

def get_context(model_path):
    with open(f"{model_path}/conversation.txt", "r") as f:
        conversation = f.readlines()
    conversation_string = "".join(conversation)
    return conversation_string

def load_gpt_model(model_path, mem_type):
    model_file = os.path.join(model_path, 'pytorch_model.bin')
    log_message(f'\nGPT model: {model_name}', '\033[37m')
    # Determine the length of the file
    file_size = os.stat(model_file).st_size
    log_message(f'Model weights size: {round(file_size/1024**3,4)} GB', '\033[37m')
    # Open the file and map it to memory
    with open(model_file, 'rb') as file:
        with mmap.mmap(-1, length=file_size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE) as mmap_obj:
            log_message(f'Loading weights from Disk to {mem_type} memory...', '\033[37m')
            st = time.time()
            mmap_obj.write(file.read())
            st = log_time(st)
            log_message(f'Extracting data from {mem_type} memory...','\033[37m')
            model_data = mmap_obj[:file_size]
            st = log_time(st)
            log_message('Loading the model state dictionary from the extracted data...','\033[37m')
            model_state_dict = torch.load(_io.BytesIO(model_data), map_location=torch.device('cpu'))
            st = log_time(st)
            log_message('Initializing the model with loaded state dictionary...','\033[37m')
            model = AutoModelForCausalLM.from_pretrained(model_path, state_dict=model_state_dict)
            st = log_time(st)
            log_message('Evaluation mode starting...','\033[37m')
            if use_cuda:
                model.to('cuda')
            else:
                model.eval()
            st = log_time(st)
            print(f'\n\033[35m==== Chat Session via {mem_type} Memory ====\033[0m\n')
            return model



class Infer(object):
    def __init__(self, args):
        self.model_name = args.model
        self.local_rank = args.local_rank
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name.replace("175b", "66b"), padding_side="left")
        self.offload_dir = os.path.abspath(os.path.expanduser(args.offload_dir))
        dtype=torch.float16
        self.args = args
        if args.int8:
            dtype=torch.int8
        dummy_weights = None
        self.model = get_ds_opt_model(self.model_name, dtype, args.cpu_offload, args.disk_offload,
            self.offload_dir, dummy_weights, args)

        prompts = ["Paris is the capital city of"]
        input_ids = self.tokenizer(
                prompts, return_tensors="pt").input_ids.cuda()

        generate_kwargs_warmup = dict(max_new_tokens=1, do_sample=False)
        with torch.no_grad():
            self.model.generate(input_ids=input_ids, **generate_kwargs_warmup)

    def run(self, prompts, gen_len=64):
        input_ids =  self.tokenizer(prompts, return_tensors='pt').input_ids.cuda()
        generate_kwargs = dict(max_new_tokens=gen_len, do_sample=False)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, **generate_kwargs)
            if self.args.local_rank > 0:
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
                     dummy_weights, args):
    import deepspeed
    import torch.distributed as dist
    from transformers.deepspeed import HfDeepSpeedConfig

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = True 

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
        # "tensor_parallel": {"tp_size": WORLD_SIZE},
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

def predict(text,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,):
    if text=="":
        yield chatbot,history,"Empty context."
        return 
    try:
        model
    except:
        yield [[text,"No Model Found"]],[],"No Model Found"
        return

    inputs = generate_prompt_with_history(text,history,tokenizer,max_length=max_context_length_tokens)
    if inputs is None:
        yield chatbot,history,"Input too long."
        return 
    else:
        prompt,inputs=inputs
        begin_length = len(prompt)
    input_ids = inputs["input_ids"][:,-max_context_length_tokens:].to(device)
    torch.cuda.empty_cache()
    global total_count, args
    total_count += 1
    print(total_count)
    if total_count % 50 == 0 :
        os.system("nvidia-smi")
    out_tok_cnt = 0
    with torch.no_grad():
        start_time = time.time()
        for x in greedy_search(input_ids,model,tokenizer,stop_words=["[|Human|]", "[|Assistant|]"],max_length=max_length_tokens,temperature=temperature,top_p=top_p):
            if is_stop_word_or_prefix(x,["[|Human|]", "[|Assistant|]"]) is False:
                if "[|Human|]" in x:
                    x = x[:x.index("[|Human|]")].strip()
                if "[|Assistant|]" in x:
                    x = x[:x.index("[|Assistant|]")].strip() 
                x = x.strip()   
                a, b=   [[y[0],convert_to_markdown(y[1])] for y in history]+[[text, convert_to_markdown(x)]],history + [[text,x]]
                out_tok_cnt += 1
                yield a, b, "Generating..."
            if shared_state.interrupted:
                shared_state.recover()
                try:
                    out_tok_cnt += 1
                    yield a, b, "Stop: Success"
                    return
                except:
                    out_tok_cnt -= 1
                    pass
        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = (out_tok_cnt ) / elapsed_time
        
    del input_ids
    gc.collect()
    torch.cuda.empty_cache()
    #print(text)
    #print(x)
    #print("="*80)
    try:
        
        yield a,b,f"Generate: Success \nMax length tokens: {max_length_tokens}, \
        Decode throughput: {tokens_per_second} tokens/seconds, \
        \nModel: {args.model} \
        \nCPU Offload: {args.cpu_offload}, Disk Offload: {args.disk_offload} \
        \nGPU Type: {torch.cuda.get_device_name(device=None)}, GPUs Used: {args.num_gpus}, Pin Memory: {1==int(args.pin_memory)}"
        
    except:
        pass
        
        
def retry(
        text,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
        ):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, f"Empty context"
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(inputs,chatbot,history,top_p,temperature,max_length_tokens,max_context_length_tokens):
        yield x



if __name__ == "__main__":
    gr.close_all()
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
    global args
    args = parser.parse_args()
  
    
    ## Hugging Face Inference
    # using local with mmap
    # model = load_gpt_model(model_path, 'mem_type')
    # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    
    ## Using .cache
    # model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    
    ## Flexgen Inference
    hf_infer = Infer(args)
    model = hf_infer.model
    tokenizer = hf_infer.tokenizer
    
    
    
    
    total_count = 0
    
    gr.Chatbot.postprocess = postprocess
    with open("assets/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()
    with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
        # available options:
        # small_and_beautiful_theme
        # darkdefault
        # gr.themes.Glass()
        # gr.themes.Soft()
        
        
        history = gr.State([])
        user_question = gr.State("")
        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)
        with gr.Row(scale=1).style(equal_height=True):
            with gr.Column(scale=5):
                with gr.Row(scale=1):
                    chatbot = gr.Chatbot(elem_id="lightelligence_chatbot").style(height="100%")
                with gr.Row(scale=1):
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(
                            show_label=False, placeholder="Enter text"
                        ).style(container=False)
                    with gr.Column(min_width=70, scale=1):
                        submitBtn = gr.Button("Send")
                    with gr.Column(min_width=70, scale=1):
                        cancelBtn = gr.Button("Stop")
                with gr.Row(scale=1):
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn") 
            with gr.Column():
                with gr.Column(min_width=50, scale=1):
                    with gr.Tab(label="Parameter Setting"):
                        gr.Markdown("# Parameters")
                        top_p = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        max_length_tokens = gr.Slider(
                            minimum=0,
                            maximum=512,
                            value=80,
                            step=8,
                            interactive=True,
                            label="Max Generation Tokens",
                        )
                        max_context_length_tokens = gr.Slider(
                            minimum=0,
                            maximum=4096,
                            value=2048,
                            step=128,
                            interactive=True,
                            label="Max History Tokens",
                        )
        gr.Markdown(description)

        predict_args = dict(
            fn=predict,
            inputs=[
                user_question,
                chatbot,
                history,
                top_p,
                temperature,
                max_length_tokens,
                max_context_length_tokens,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )
        retry_args = dict(
            fn=retry,
            inputs=[
                user_input,
                chatbot,
                history,
                top_p,
                temperature,
                max_length_tokens,
                max_context_length_tokens,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )

        reset_args = dict(
            fn=reset_textbox, inputs=[], outputs=[user_input, status_display]
        )
    
        # Chatbot
        transfer_input_args = dict(
            fn=transfer_input, inputs=[user_input], outputs=[user_question, user_input, submitBtn], show_progress=True
        )

        predict_event1 = user_input.submit(**transfer_input_args).then(**predict_args)

        predict_event2 = submitBtn.click(**transfer_input_args).then(**predict_args)

        emptyBtn.click(
            reset_state,
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )
        emptyBtn.click(**reset_args)

        predict_event3 = retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            [chatbot, history, status_display],
            show_progress=True,
        )
        cancelBtn.click(
            cancel_outputing, [], [status_display], 
            cancels=[
                predict_event1,predict_event2,predict_event3
            ]
        )    


    demo.title = "LT-Chat"
    demo.queue(concurrency_count=1).launch(server_port=9808,server_name='10.102.128.22',share=True)
    # demo.queue(concurrency_count=1).launch(server_port=9808,server_name='10.102.128.22',share=True)