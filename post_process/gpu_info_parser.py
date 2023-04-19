# coding=utf8

import os
import json
import argparse


def main(args):
    dir_path = args.input
    out_path = args.output
    device_num = args.device
    res = dict()
    f_names = os.listdir(dir_path)
    end_suffix = '.txt.gpu_mem.json'
    for f_name in f_names:
        if f_name.endswith(end_suffix):
            f_path = os.path.join(dir_path, f_name)
            max_util = get_info(f_path, device_num)
            if max_util is not None:
                key = f_name.rstrip(end_suffix)
                res[key] = dict()
                res[key]['max_gpu_mem'] = max_util

    if out_path:
        with open(out_path, 'w') as out_:
            json.dump(res, out_)
    return res


def get_info(f_path, device_num):
    max_util = 0
    with open(f_path) as in_:
        i = 0
        for line in in_:
            i = 1+i;
            line = line.strip()
            if len(line) > 0:
                try:
                    obj = json.loads(line)
                    _arr = obj['gpu']
                    elem = _arr[device_num]['fb_memory_usage']
                    total = elem['total']
                    free = elem['free']
                    used = total - free
                    if used > max_util:
                        max_util = used
                except Exception as e:
                    print(f"error in parsing line {i}")
                    print(e)
                    print(line)
    return max_util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    t_args = parser.parse_args()
    main(t_args)

