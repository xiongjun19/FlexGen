log_dir=/workspace/data/flex_tp_logs_v2/
gpu_logs=${log_dir}/gpu_logs
mkdir -p ${gpu_logs}


set -x 

# tp_arr=(1 2 4)
tp_arr=(1 2 4 8) 
for(( i=0;i<${#tp_arr[@]};i++)) do
   tp=${tp_arr[i]};
   gpu_log="${gpu_logs}/opt-30b_tp${tp}.qdrep";
   # CUDA_VISIBLE_DEVICE=4,6 python  tp_flex_opt.py --model facebook/opt-30b  --percent 0 100 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 128 --gen-len 8 --tp_num $tp --path _DUMMY_;
   # CUDA_VISIBLE_DEVICE=4,6 nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} python  tp_flex_opt_prof.py --model facebook/opt-30b  --percent 0 100 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 128 --gen-len 8 --tp_num $tp --path _DUMMY_;
   echo "finished one"
done
 

tp_arr=(1 2 4 8) 
for(( i=0;i<${#tp_arr[@]};i++)) do
   tp=${tp_arr[i]};
   gpu_log="${gpu_logs}/opt-66b_tp${tp}.qdrep";
   CUDA_VISIBLE_DEVICE=4,6 python  tp_flex_opt.py --model facebook/opt-66b  --percent 0 100 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 128 --gen-len 8 --tp_num $tp --path _DUMMY_;
   CUDA_VISIBLE_DEVICE=4,6 nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} python  tp_flex_opt_prof.py --model facebook/opt-66b  --percent 0 100 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 128 --gen-len 8 --tp_num $tp --path _DUMMY_;
   echo "finished one"
done
 
