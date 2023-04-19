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


def main(args):
    dir_path = args.input
    out_path = args.output
    gpu_info_config = ConfigInfo(os.path.join(dir_path, 'gpu_logs'), 'out_path_tmp_nsys.json', args.device, args.times)
    cpu_info_config = ConfigInfo(os.path.join(dir_path, 'cpu_logs'), 'out_path_tmp_cpu.json', args.device, args.times)
    nsys_dict = nsys_batch_parser.main(gpu_info_config)
    lat_th_dict = latency_and_th_parser.main(cpu_info_config)
    gpu_mem_dict = gpu_info_parser.main(cpu_info_config)
    cpu_info_dict = cpu_info_parser.main(cpu_info_config)
    sheet_name = args.sheet_name
    _merge_and_save(nsys_dict, lat_th_dict, gpu_mem_dict, cpu_info_dict, out_path, sheet_name)


def _merge_and_save(nsys_dict, lat_th_dict, gpu_mem_dict, cpu_info_dict, out_path, sheet_name):
    df = _merge_to_df(nsys_dict, lat_th_dict, gpu_mem_dict, cpu_info_dict)
    df.to_excel(out_path, index=False)
    # wb = Workbook()
    # with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    #     if os.path.exists(out_path):
    #         writer.book = openpyxl.load_workbook(out_path)
    #     else:
    #         writer.book = wb
    #     writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)
    #     df.to_excel(writer, sheet_name=sheet_name)
    #     writer.save()


def _merge_to_df(nsys_dict, lat_th_dict, gpu_mem_dict, cpu_info_dict):
    tot_dict = {}
    _merge_impl(tot_dict, nsys_dict)
    _merge_impl(tot_dict, lat_th_dict)
    _merge_impl(tot_dict, gpu_mem_dict)
    _merge_impl(tot_dict, cpu_info_dict)
    df = _cvt_to_df(tot_dict)
    return df


def _cvt_to_df(_dict):
    '''
    opt-66b_1024_4
    {'tot': 29888.157887, 'util': 0.03079819532472346, 'nccl': 0.0, 'mem': 0.9699826427793682, 'prefill_lat': 6.525, 'dec_lat': 0.154, 'tot_lat': 19.518, 'tot_tho': 0.154, 'max_gpu_mem': 7061.0, 'avg_cpu_util': 100.99276923076907, 'avg_cpu_io_wait': 0.0, 'max_mem_footprint': 223983736.0, 'max_tps': 834328204.0, 'avg_tps': 834174897.3538462, 'max_bread': 115212.0, 'avg_bread': 115151.32307692307, 'max_bwrtn': 213160224.0, 'avg_bwrtn': 213148060.6153846}
    '''
    key_map = {
            'util': 'GPU_Util',
            'mem': 'io_time_ratio',
            'tot_tho': 'throughput(tokens/s)',
            'max_mem_footprint': 'max_cpu_mem(KB)',
            'max_gpu_mem': 'max_gpu_mem(MB)',
            }
    res = {
            'model_param': [],
            'batch_size': [],
            'num_of_batch': [],
            'input_len': [],
            'output_len': [],
            'percent': [],
            "compression": [],
          }
    for key, val_dict in _dict.items():
        key_info = _extract_key(key)
        tuples = zip(['model_param', 'input_len', 'output_len', 'batch_size', 'num_of_batch', 'percent', 'compression'], key_info)
        for x, y in tuples:
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
    # import ipdb; ipdb.set_trace()
    df = pd.DataFrame.from_dict(res)
    return df


def _extract_key(key):
    _arr = key.split("_")
    param = _arr[0]
    input_len = _arr[1]
    output_len = _arr[2]
    bs = 1
    num_gpu = 1
    per = None
    compression = None
    if len(_arr) > 4:
        bs = _arr[3]
        num_gpu = _arr[4]
    if len(_arr) > 6:
        per = _arr[5]
        compression = _arr[6]
    return param, input_len, output_len, bs, num_gpu, per, compression

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

