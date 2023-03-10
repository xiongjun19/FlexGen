# coding=utf8


import time
import json
from pynvml.smi import nvidia_smi


def main(out_path):
    with open(out_path, 'w') as out_:
        nvsmi = nvidia_smi.getInstance()
        while True:
            res = nvsmi.DeviceQuery('memory.free, memory.total')
            json_str = json.dumps(res)
            out_.write(json_str + "\n")
            time.sleep(1)


if __name__ == '__main__':
    import sys
    f_path = sys.argv[1]
    main(f_path)
