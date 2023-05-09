
import json


def do_parse(f_path, out_path):
    mem_info = _parse_mem(f_path)
    save_mem_info(mem_info, out_path)


def _parse_mem(f_path):
    start = False
    res_dict = {}
    print("path name is: ", f_path)
    with open(f_path, 'rb') as in_:
        for line in in_:
            try:
                line = line.decode('utf-8')
            except:
                print('invalid value')
                continue
            line = line.strip()
            if "benchmark - generate" in line:
                start = True
            elif start:
                elem_info = _parse_line(line)
                if elem_info is not None:
                    copy_kind, _bytes = elem_info
                    if copy_kind not in res_dict:
                        res_dict[copy_kind] = {}
                    _append_to_dict(res_dict[copy_kind], 'bytes', _bytes * 2 / 1024) # convert bytes to KB
    return res_dict


def _parse_line(line):
    if "Disk" not in line:
        return None
    line_arr = line.split('%')
    if len(line_arr) == 2:
        op = line_arr[0]
        size = line_arr[1].split(':')[-1].strip()
        if size.isdigit():
            return op, int(size)
    return None


def _append_to_dict(_dict, key, val):
    if key not in _dict:
        _dict[key] = []
    _dict[key].append(val)


def test(f_path, o_path):
    res = do_parse(f_path, o_path)
    print("done")


def save_mem_info(mem_info, out_path):
    with open(out_path, 'w') as _out:
        json.dump(mem_info, _out)


if __name__ == '__main__':
    import sys
    t_path = sys.argv[1]
    o_path = sys.argv[2]
    test(t_path, o_path)
