# coding=utf8

import flask
import hf_infer

app = flask.Flask(__name__)
app.config["DEBUG"]

infer = None

def init_predictor(conf):
    global infer
    infer = hf_infer.Infer(conf)

@app.route('/gen_cpu_off', methods=['POST', 'GET'])
def gen_info():
    prompts_info = request.json
    prompts = prompts_info.get('prompts')
    max_len = prompts_info.get('max_len', 8)
    if prompts is None:
        result = {
            "status": 1,
            "msg": "没有解析到prompts, 请检查传入的参数"
        }
        return json.dumps(result, ensure_ascii=False)

    outputs = infer.run(prompts, max_len)
    result = {
        "status": 0,
        "msg": outputs
    }
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dummy", action="store_true",
        help="Use dummy weights for benchmark purposes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--disk-offload", action="store_true")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir")
    parser.add_argument("--int8", action="store_true")

    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--pkl-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args()
    init_predictor(args)
    app.run('0.0.0.0', port='9808')

