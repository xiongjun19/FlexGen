# coding=utf8

import time
import os
import argparse
import shlex
import subprocess
import json
from functools import wraps
from functools import partial




def get_cur_time():
    local_time = time.localtime()
    time_str = time.strftime('%H:%M:%S', local_time)
    return time_str

def time_log(func):
    def wrapper(*args,  **kwargs):
        cur_time = get_cur_time()
        print("starting to running")
        print("start time is: {0}".format(cur_time))
        st = time.time()
        res = func(*args, **kwargs)
        print("finished running, time consumed is: {0}".format(time.time() - st))
        end_time = get_cur_time()
        print("finished time is: {0}".format(end_time))
        return res
    return wrapper


def exec_cmd(cmd_str):
    # cmd_args = shlex.split(cmd_str)
    # subprocess.call(cmd_args)
    print("shell cmd is: {0}".format(cmd_str))
    subprocess.call(cmd_str, shell=True)


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def perf_log(log_path=None):
    def decorate(func):
        global out_path
        out_path = log_path if log_path else '/tmp/sar_log.txt'

        @wraps(func)
        def wrapper(*args, **kwargs):
            dir_path = os.path.dirname(out_path)
            pre_cmd = 'mkdir -p {0}; nohup sar -buqp  -r ALL -n DEV -P ALL  1 > {1} &'.format(dir_path, out_path)
            exec_cmd(pre_cmd)
            abs_path , _ = os.path.split(__file__) 
            gpu_util_path = os.path.join(abs_path, "get_gpu_util.py")
            pre_cmd = 'nohup python {0} {1} &'.format(gpu_util_path, out_path + ".gpu_mem.json")
            exec_cmd(pre_cmd)
            time.sleep(1)
            res = func(*args, **kwargs)
            time.sleep(1)
            after_cmd = 'ps aux | grep sar | grep buqp | awk \'{print $2}\' | xargs kill -9'
            exec_cmd(after_cmd)
            after_cmd = 'ps aux | grep python | grep get_gpu_util | awk \'{print $2}\' | xargs kill -9'
            exec_cmd(after_cmd)
            time.sleep(1)
            return res

        @attach_wrapper(wrapper)
        def set_path(f_path):
            global out_path
            out_path = f_path

        return wrapper

    return decorate


class SqlProfiler(object):
    def __init__(self, args):
        self.config = args
        self.res_dir = self.config.res
        if not os.path.exists(self.res_dir):
            # os.makedirs(self.res_dir, exist_ok=True)
            os.makedirs(self.res_dir)
        self.perf_log_dir = self.config.perf_log
        if not os.path.exists(self.perf_log_dir):
            os.makedirs(self.perf_log_dir)

    @perf_log('/tmp/tmp_log.txt')
    def run_sql(self, f_path):
        db = self.config.db
        b_name = os.path.basename(f_path)
        res_path = os.path.join(self.res_dir, b_name)
        cmd = "psql -h localhost -U {0} {1} < {2} > {3}".format(USER, db, f_path, res_path)
        # cmd = "sed '$d' {0}".format(f_path)
        cur_time = get_cur_time()
        print("starting to running")
        print("start time is: {0}".format(cur_time))
        st = time.time()
        exec_cmd(cmd)
        cost_time = time.time() - st
        print("finished running, time consumed is: {0}".format(cost_time))
        end_time = get_cur_time()
        print("finished time is: {0}".format(end_time))
        return cur_time, end_time, cost_time


    def profile(self, sql_dir):
        time_dict = dict()
        f_arr = os.listdir(sql_dir)
        f_arr = [os.path.join(sql_dir, x) for x in f_arr if x.endswith(".sql")]
        for f_path in f_arr:
            f_b_name = os.path.basename(f_path)
            print("starting to run ", f_path)
            perf_log_path = os.path.join(self.perf_log_dir, f_b_name)
            self.run_sql.set_path(perf_log_path)
            st_time, end_time, cus_time = self.run_sql(f_path)
            time_dict[f_b_name] = {
                    'begin': st_time,
                    'end': end_time,
                    'time': cus_time}
            print("finished running")
        time_path = os.path.join(self.res_dir, 'time.sum.txt')
        with open(time_path, 'w') as out_:
            json.dump(time_dict, out_)

        print("done")


@time_log
def test_print(test_str):
    time.sleep(1)
    print("echo str: ", test_str)


def test(args):
    test_print("hello world")


def main(args):
    profiler = SqlProfiler(args)
    profiler.profile(args.in_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, help='file path to the directory')
    parser.add_argument('-o', '--sar_log', type=str, help='path to the log of sar')
    parser.add_argument('-d', '--db', type=str, default='tpch_200g', help='database name')
    parser.add_argument('-r', '--res', type=str, default='results', help='directory to store the results')
    parser.add_argument('-p', '--perf_log', type=str, default='/data/perf_log', help='directory path of the performance log')
    args = parser.parse_args()
    # test(args)
    main(args)

