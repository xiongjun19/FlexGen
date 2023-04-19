# coding=utf8

import os
import json
import argparse


def main(args):
    dir_path = args.input
    out_path = args.output
    res = dict()
    f_names = os.listdir(dir_path)
    end_suffix = '.txt'
    for f_name in f_names:
        if f_name.endswith(end_suffix):
            f_path = os.path.join(dir_path, f_name)
            info = get_info(f_path)
            if info is not None:
                key = f_name.rstrip(end_suffix)
                res[key] = info

    if out_path:
        with open(out_path, 'w') as out_:
            json.dump(res, out_)
    return res


def get_info(f_path):
    max_util = 0
    res = dict()
    avg_cpu_util, avg_cpu_io_wait =  parse_cpu_info(f_path)
    res['avg_cpu_util'] = avg_cpu_util
    res['avg_cpu_io_wait'] = avg_cpu_io_wait
    max_mem_footprint =  parse_mem_info(f_path)
    res['max_mem_footprint'] = max_mem_footprint
    max_tps, avg_tps, max_bread, avg_bread, max_bwrtn, avg_bwrtn = parse_disk_info(f_path)
    res['max_tps'] = max_tps
    res['avg_tps'] = avg_tps
    res['max_bread'] = max_bread
    res['avg_bread'] = avg_bread
    res['max_bwrtn'] = max_bwrtn
    res['avg_bwrtn'] = avg_bwrtn
    return res


    with open(f_path) as in_:
        for line in in_:
            line = line.strip()
            if len(line) > 0:
                obj = json.loads(line)
                _arr = obj['gpu']
                elem = _arr[device_num]['fb_memory_usage']
                total = elem['total']
                free = elem['free']
                used = total - free
                if used > max_util:
                    max_util = used
    return max_util


def parse_cpu_info(f_path):
    "avg_cpu usage, io_wait"
    _sum = 0
    cnt_num = 0
    io_wait_sum = 0
    b_st = False
    cu_st = False
    with open(f_path) as in_:
        for line in in_:
            cu_st = _get_cpu_stat(line, b_st)
            if cu_st and b_st:
                _usage, io_wait = _parse_cpu_detail(line)
                io_wait_sum += io_wait
                _sum += _usage
            if not cu_st and b_st:
                cnt_num += 1
            b_st = cu_st
        if cu_st and  b_st:
            cnt_num += 1
    if cnt_num == 0:
        return 0., 0.
    avg_io_wait = io_wait_sum / cnt_num
    avg_cpu_util = _sum / cnt_num
    return avg_cpu_util, avg_io_wait


def _get_cpu_stat(line, stat):
    line = line.strip()
    line_arr = line.split()
    if len(line_arr) < 2:
        return False
    sign = line_arr[1]
    if sign == 'CPU':
        return True
    if stat:
        if sign.lower() == 'all' or sign.isdigit():
            return True
    return False


def _parse_cpu_detail(line):
    line = line.strip()
    line_arr = line.split()
    sign = line_arr[1].lower()
    if sign == 'all':
        return 0., float(line_arr[5])
    return float(line_arr[2]) + float(line_arr[3]) + float(line_arr[4]), 0.


def parse_mem_info(f_path):
    max_mem = 0.
    b_st = False
    cur_st = False
    with open(f_path) as in_:
        for line in in_:
            cur_st = _get_mem_stat(line, b_st)
            if b_st:
                mem = _parse_mem_detail(line)
                if max_mem < mem:
                    max_mem = mem
            b_st = cur_st
    return max_mem


def _get_mem_stat(line, stat):
    line = line.strip()
    line_arr = line.split()
    if len(line_arr) < 2:
        return False
    sign = line_arr[1]
    if 'memfree' in sign.lower():
        return True
    return False


def _parse_mem_detail(line):
    line = line.strip()
    line_arr = line.split()
    return float(line_arr[7])


def parse_disk_info(f_path):
    tps_sum = 0
    max_tps = 0
    bread_sum = 0
    max_bread = 0
    bwrtn_sum = 0
    max_bwrtn = 0
    cnt = 0
    b_st = False
    cur_st = False
    with open(f_path) as in_:
        for line in in_:
            cur_st = _get_mem_stat(line, b_st)
            if b_st:
                cnt += 1
                tps, bread, bwrtn = _parse_disk_detail(line)
                tps_sum += tps
                bread_sum += bread
                bwrtn_sum += bwrtn
                if max_tps < tps:
                    max_tps = tps
                if max_bread < bread:
                    max_bread = bread
                if max_bwrtn < bwrtn:
                    max_bwrtn = bwrtn
            b_st = cur_st
    if cnt == 0:
        return 0, 0, 0, 0, 0, 0
    return max_tps, tps_sum / cnt, max_bread, bread_sum / cnt, max_bwrtn, bwrtn_sum / cnt


def _get_disk_stat(line, stat):
    line = line.strip()
    line_arr = line.split()
    if len(line_arr) < 2:
        return False
    sign = line_arr[1]
    if 'tps' == sign.lower():
        return True
    return False


def _parse_disk_detail(line):
    line = line.strip()
    line_arr = line.split()
    return float(line_arr[1]), float(line_arr[5]), float(line_arr[6])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    t_args = parser.parse_args()
    main(t_args)

