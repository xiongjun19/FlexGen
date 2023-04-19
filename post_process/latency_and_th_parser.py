# coding=utf8

import os
import json
import argparse


def main(args):
    dir_path = args.input
    out_path = args.output
    res = dict()
    f_names = os.listdir(dir_path)
    end_suffix = '.txt.org'
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
    with open(f_path) as in_:
        lines = in_.readlines()
        if len(lines) >= 5:
            prefill_line = lines[2]
            dec_line = lines[3]
            tot_line = lines[4]
            pref_lat, pref_tho = _parse_line(prefill_line)
            dec_lat, dec_tho = _parse_line(dec_line)
            tot_lat, tot_tho = _parse_line(tot_line)
            return {"prefill_lat": pref_lat, "dec_lat": dec_lat, "tot_lat": tot_lat, "tot_tho": tot_tho, 'gen_tho': dec_tho}
    return None


def _parse_line(line):
    line = line.strip()
    sub_lines = line.split("\t")
    lat_line = sub_lines[0]
    tho_line = sub_lines[1]
    lat = _get_digit(lat_line)
    tho = _get_digit(tho_line)
    return lat, tho


def _get_digit(line):
    _arr = line.split(":")
    digit = _arr[-1].strip().split()[0]
    return float(digit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    t_args = parser.parse_args()
    main(t_args)

