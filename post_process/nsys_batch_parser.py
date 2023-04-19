# coding=utf8

import os
import json
import argparse
from nsys_parser import do_parse


def main(args):
    dir_path = args.input
    out_path = args.output
    times = args.times * 1e+6
    res = dict()
    f_names = os.listdir(dir_path)
    for f_name in f_names:
        if f_name.endswith(".sqlite"):
            f_path = os.path.join(dir_path, f_name)
            tot_time, util, nccl_ratio, mem_ratio = do_parse(f_path)
            if tot_time is not None:
                key = f_name.rstrip(".sqlite")
                res[key] = dict()
                # res[key]['tot'] = tot_time / 4e+6
                res[key]['tot'] = tot_time / times
                res[key]['util'] = util
                res[key]['nccl'] = nccl_ratio
                res[key]['mem'] = mem_ratio

    with open(out_path, 'w') as out_:
        json.dump(res, out_)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--times', type=int, default=2)
    t_args = parser.parse_args()
    main(t_args)

