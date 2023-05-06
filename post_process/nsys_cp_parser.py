# coding=utf8


import json
import sqlite3


def do_parse(f_path, out_path):
    mydb = sqlite3.connect(f_path)
    cursor = mydb.cursor()
    mem_info = _parse_mem(cursor)
    save_mem_info(mem_info, out_path)
    cursor.close()
    mydb.close()


def _parse_mem(cursor):
    # sql = "select sum(end - start) from CUPTI_ACTIVITY_KIND_MEMSET where deviceId=0;"
    # set_time = exec_and_parse(cursor, sql)
    # sql = "select sum(end - start) from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0;"
    sql = "select end - start as time, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0;"
    res = exec_and_parse(cursor, sql)
    if res is None:
        return {}
    res_dict = {}
    for item in res:
        time = item[0]
        _bytes = item[1]
        copy_kind = item[2]
        if copy_kind not in res_dict:
            res_dict[copy_kind] = {}
        _append_to_dict(res_dict[copy_kind], 'time', time / 1000) # convert ns to micro seconds
        _append_to_dict(res_dict[copy_kind], 'bytes', _bytes / 1024) # convert bytes to KB
    return res_dict


def _append_to_dict(_dict, key, val):
    if key not in _dict:
        _dict[key] = []
    _dict[key].append(val)


def exec_query(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()


def exec_and_parse(cursor, sql):
    try:
        cursor.execute(sql)
        items = cursor.fetchall()

        return items
    except sqlite3.OperationalError as e:
        print(e)
        return None


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
