# coding=utf8

import os
import json
import argparse
import pandas as pd
from dataclasses import dataclass
import openpyxl
from openpyxl import Workbook
import gpu_info_parser
import latency_and_th_parser
import nsys_batch_parser
import cpu_info_parser


@dataclass
class ConfigInfo:
    input: str
    output: str
    device: int = 0
    times: int = 2
    card_name: str


def main(args):
    dir_path = args.input
    out_path = args.output
    cpu_info_config = ConfigInfo(dir_path, 'out_path_tmp_cpu.json', args.device, args.times, args.card_name)
    lat_th_dict = latency_and_th_parser.main(cpu_info_config, '.txt')
    sheet_name = args.sheet_name
    _merge_and_save(lat_th_dict, out_path, sheet_name, card_name)


def _merge_and_save(lat_th_dict, out_path, sheet_name, card_name):
    df = _merge_to_df(lat_th_dict, card_name)
    df.to_excel(out_path, index=False)


def _merge_to_df(lat_th_dict, card_name):
    df = _cvt_to_df(lat_th_dict, card_name)
    return df


def _cvt_to_df(_dict, card_name):
    key_map = {
            'util': 'GPU_Util',
            'mem': 'io_time_ratio',
            'tot_tho': 'throughput(tokens/s)',
            'max_mem_footprint': 'max_cpu_mem(KB)',
            'max_gpu_mem': 'max_gpu_mem(MB)',
            }
    res = {}
    key_arr = ['model_name', 'parallel', 'input_len', 'output_len',
               'bs', 'num_bs', 'compression', 'policy', 'card_num', ]
    for key, val_dict in _dict.items():
        key_info = _extract_key(key)
        tuples = zip(key_arr, key_info)
        for x, y in tuples:
            if x not in res:
                res[x] = []
            res[x].append(y)

        for k, val in val_dict.items():
            if not _is_keep(k):
                continue
            new_key = k
            if k in key_map:
                new_key = key_map[k]
            val = round(val, 3)
            if new_key not in res:
                res[new_key] = []
            res[new_key].append(val)
    card_arr = [card_name] * len(res['model_name'])
    res['card_name'] = card_arr
    df = pd.DataFrame.from_dict(res)
    return df


def _extract_key(key):
    _arr = key.split("_")
    model_name = _arr[0]
    parallel = _arr[1]
    input_len = _arr[2]
    output_len = _arr[3]
    bs = _arr[4]
    num_bs = _arr[5]
    compression = _arr[6]
    policy = _arr[7]
    card_num = _arr[8]
    return model_name, parallel, input_len, output_len, \
           bs, num_bs, compression, policy, card_num

def _is_keep(key):
    if key == 'tot':
        return False
    return True


def _merge_impl(tot_dict, new_dic):
    for k, v in new_dic.items():
        if k in tot_dict:
            tot_dict[k].update(v)
        else:
            tot_dict[k] = v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='the log path')
    parser.add_argument('--output', type=str, default=None, help='the output excel file')
    parser.add_argument('--device', type=int,  default=0)
    parser.add_argument('--times', type=int,  default=2)
    parser.add_argument('--sheet_name', type=str, default='sheet1')
    t_args = parser.parse_args()
    main(t_args)

