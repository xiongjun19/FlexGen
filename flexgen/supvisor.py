# coding=utf8

import time
import json
import os
import argparse
import subprocess
import shlex
from subprocess import Popen


def exe_cmd_sync(cmd_line):
    # call is blocking:
    cmd_args = shlex.split(cmd_line)
    subprocess.call(cmd_args)


def kill_process(key_word):
    cmd = f"ps aux | grep flex | grep opt | grep {key_word} | " 
    cmd_part2 = 'awk \'{print $2}\' | xargs kill -9'
    cmd_line = cmd + cmd_part2
    exe_cmd_sync(cmd_line)


def get_lines(in_file, line_no):
    res = []
    with open(in_file) as _in:
        num = 0
        for line in _in:
            if num >= line_no:
                res.append(line)
            num += 1
    return res


def get_keyword(lines):
    start_no = -1
    key = None
    num = 0
    for line in lines:
        if "starting collect file" in line:
            start_no = num
            key = line.strip().split(":")[1].strip()
    return key, start_no


def _need_kill(lines, key, start_no):
    if key is None:
        return False
    num = 0
    for line in lines:
        if num > start_no:
            if 'error' in line.lower():
                return True
    return False


def check_and_kill(in_file, line_no):
    lines = get_lines(in_file, line_no)
    line_no += len(lines)
    key_word, start_no = get_keyword(lines)
    if _need_kill(lines, key_word, start_no):
        kill_process(key_word)
    return line_no


def main(in_file):
    line_no = 0
    while True:
        line_no = check_and_kill(in_file, line_no)
        time.sleep(2)


if __name__ == '__main__':
    import sys
    input_file = sys.argv[1]
    main(input_file)
