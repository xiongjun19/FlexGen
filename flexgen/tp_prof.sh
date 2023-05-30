log_dir=/workspace/data/flex_tp_logs/
gpu_logs=${log_dir}/gpu_logs
mkdir -p ${gpu_logs}



tp_arr=(1 2 4)
for(( i=0;i<${#tp_arr[@]};i++)) do
   tp=${tp_arr[i]};
   gpu_log="${gpu_logs}/$opt-30b_tp${tp}.qdrep";
   nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} python  tp_flex_opt_prof.py --model facebook/opt-30b  --percent 0 100 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 128 --gen-len 8 --tp_num $tp;
   echo "finished one"
done
 
