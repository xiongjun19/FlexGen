# coding=utf8

import os
import time
import json
import argparse


day_format = '%Y%m%d'
sec_format = '%H:%M:%S'
complete_format = " ".join([day_format, sec_format])


def mk_dir(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def load_time_dict(f_path):
    with open(f_path) as in_:
        res = json.load(in_)
        return res


def clean_dir(in_dir, out_dir, f_time_sum):
    f_time_dict = load_time_dict(f_time_sum)
    for root, dir_names, f_names in os.walk(in_dir):
        dst_dir = _cvt_dst_dir(root, in_dir, out_dir)
        mk_dir(dst_dir)
        for f_name in f_names:
            if f_name in f_time_dict:
                cur_dict = f_time_dict[f_name]
                st_time = cur_dict['begin']
                end_time = cur_dict['end']
                in_file = os.path.join(root, f_name)
                out_file = os.path.join(dst_dir, f_name)
                clean_by_time(in_file, out_file, st_time, end_time)


def _cvt_dst_dir(src_dir, in_dir, out_dir):
    src_dir = src_dir.rstrip("/")
    in_dir = in_dir.rstrip("/")
    if len(out_dir) > 1:
        out_dir = out_dir.rstrip("/")
    if src_dir == in_dir:
        return out_dir
    rel_dir = src_dir[len(in_dir)+1:]
    res = os.path.join(out_dir, rel_dir)
    return res


def clean_by_time(f_path, out_path, start_time, end_time):
    with open(f_path) as in_:
        with open(out_path, 'w') as out_:
            line_arr = in_.readlines()
            first_line = line_arr[0]
            out_.write(first_line)
            time_arr = _get_time_arr(line_arr)
            bg_idx = _get_index(time_arr, start_time, -1)
            end_idx = _get_index(time_arr, end_time, 1)
            if bg_idx and end_idx:
                new_line_arr = line_arr[bg_idx:end_idx]
                for line in new_line_arr:
                    out_.write(line)


def _get_time_arr(line_arr):
    res = []
    for i, line in enumerate(line_arr):
        seg_arr = line.split()
        if len(seg_arr) > 1:
            if seg_arr[1] == 'CPU':
                res.append([seg_arr[0], i])
    return res


def _get_index(time_arr, dst_time, next_offset=0):
    for j, tup_elem in enumerate(time_arr):
        if dst_time == tup_elem[0]:
            idx = j + next_offset
            res = time_arr[idx][1]
            return res
    return None


def convert_time(time_str, today, sec=0):
    comp_str = add_today_prefix(today, time_str)
    if sec == 0:
        return comp_str
    new_str = add_secs(comp_str, sec)
    res = add_today_prefix(today, new_str)
    return res


def get_today(time_format=day_format):
    localtime = time.localtime()
    return time.strftime(time_format, localtime)


def add_today_prefix(today, time_str):
    time_str = time_str.strip()
    time_arr = time_str.split(" ")
    sec_part = time_arr[-1]
    res = " ".join([today, sec_part])
    return res


def add_secs(time_str, sec=1, time_format=complete_format):
    timestamp = time.mktime(time.strptime(time_str, time_format))
    new_timestamp = timestamp + sec
    res = time.strftime(time_format, time.localtime(new_timestamp))
    return res


def test():
    import sys
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    st_time = '14:34:37'
    end_time = '14:34:43'
    clean_by_time(in_file, out_file, st_time, end_time)


def main(args):
    clean_dir(args.input,
              args.output,
              args.time_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='dir path of the input data')
    parser.add_argument('-o', '--output', type=str, help='dir path of the output data')
    parser.add_argument('-t', '--time_path', type=str, help='path to the time summary file')
    args = parser.parse_args()
    main(args)

