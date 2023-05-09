
import os
import argparse
import disk_read_write_parser


def main(args):
    dir_path = args.input
    out_path = args.output
    os.makedirs(out_path, exist_ok=True)
    f_arr = os.listdir(dir_path)
    for f in f_arr:
        if f.endswith("org_data_trans.txt"):
            f_path = os.path.join(dir_path, f)
            out_file = os.path.join(out_path, f)
            disk_read_write_parser.do_parse(f_path, out_file)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='the log path')
    parser.add_argument('--output', type=str, default=None, help='the output excel file')
    parser.add_argument('--device', type=int,  default=0)
    parser.add_argument('--times', type=int,  default=2)
    parser.add_argument('--sheet_name', type=str, default='sheet1')
    t_args = parser.parse_args()
    main(t_args)

