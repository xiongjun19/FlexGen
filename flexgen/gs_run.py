# coding=utf8


import os
import argparse
import subprocess
import shlex
from tqdm import tqdm
from subprocess import Popen


def exe_cmd_sync(cmd_line):
    # call is blocking:
    cmd_args = shlex.split(cmd_line)
    subprocess.call(cmd_args)


def iter_recursive(arr_list):
    depth_idx = 0
    # res = []
    arr_num = len(arr_list)
    tmp_arr = [None] * arr_num
    width_idx_arr = [0] * arr_num
    while depth_idx >= 0:
        width_idx = width_idx_arr[depth_idx]
        if width_idx >= len(arr_list[depth_idx]):
            width_idx_arr[depth_idx] = 0
            depth_idx = depth_idx - 1
            width_idx_arr[depth_idx] += 1
            continue
        tmp_arr[depth_idx] = arr_list[depth_idx][width_idx]
        if depth_idx < arr_num - 1:
            depth_idx += 1
        else:
            yield(tmp_arr.copy())
            width_idx_arr[depth_idx] += 1


def get_para_arrs():
    bs_arr = [1, 4, 16, 32, 64, 128]
    num_bs_arr = [1, 2, 4, 16]
    weight_arr = ["0:100", "10:90", "20:80", "30:70"]
    cache_arr = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]
    act_arr = ["100:0", "0:100"]
    comp_arr = [0, 1]
    res_arr = [
            bs_arr, num_bs_arr,
            weight_arr, cache_arr, act_arr,
            comp_arr,
    ]
    return res_arr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ig', type=int, default=1, help='num_of_gpu')
    parser.add_argument('--para', type=str, default='no-para', help='number of gpu')
    parser.add_argument('--out', type=str, default='/workspace/data/gs_res')
    args = parser.parse_args()
    return args


def main():
    off_args = "--offload-dir /workspace/data/flex_offload_dir"
    args = get_args()
    card_num = 1
    exec_f = 'tp_flex_opt.py'
    pipe_mode = False
    tp_mode = False
    if args.para == 'pp' or args.para == 'pipe':
        pipe_mode = True
        exec_f = 'dist_flex_opt.py'
        card_num = args.ig
    elif args.para == 'tp' or args.para == 'tensor-para':
        tp_mode = True
        card_num = args.ig
        exec_f = 'tp_flex_opt.py'
    os.makedirs(args.out, exist_ok=True)
    model_name = 'opt-66b'
    context_len = 512
    output_len = 8
    para_list = iter_recursive(get_para_arrs())
    for bs, num_bs, weight_po, cache_po, act_po, comp in tqdm(para_list):
        policy = ":".join([weight_po, cache_po, act_po])
        act_bs = bs
        if pipe_mode:
            act_bs = bs * 2
        f_path = _get_log_path(model_name, pipe_mode, tp_mode, context_len,
                               output_len, act_bs, num_bs, policy, comp, card_num)
        f_path = os.path.join(args.out, f_path)
        policy = policy.replace(":", " ")

        cmd_args = f"{exec_f} --model facebook/{model_name} {off_args} " \
                f"--path _DUMMY_ --percent {policy} --gpu-batch-size {bs} " \
                f"--num-gpu-batches {num_bs}  --prompt-len {context_len} --gen-len {output_len} " \
                f"--log-file {f_path}"
        if not pipe_mode:
            cmd = f"python {cmd_args} --tp {card_num} "
        else:
            cmd = f" mpirun --allow-run-as-root --map-by ppr:{card_num}:node:pe=12 "\
                  f" --oversubscribe --bind-to core -x OMP_NUM_THREADS=12 " \
                  f"python {cmd_args} --use-mpi  --comm-device cpu  "
        if comp == 1:
            cmd = cmd + " " + "--compress-weight --compress-cache"
        exe_cmd_sync(cmd)



def _get_log_path(model_name, pipe_mode, tp_mode,
    context_len, out_len, bs, num_bs,
    policy, comp, card_num):
    para='none'
    if pipe_mode:
        para='pp'
    if tp_mode:
        para='tp'
    path_name = f"{model_name}_{para}_{context_len}_{out_len}_{bs}_{num_bs}_{comp}_{policy}_{card_num}.txt"
    return path_name


if __name__ == '__main__':
    main()
