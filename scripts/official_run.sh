log_dir=/workspace/data/flex_logs_2
mkdir -p ${log_dir}
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}

model_prefix="facebook"
model_name_arr=("opt-175b" "opt-175b")
input_len_arr=(512 1024)
out_len_arr=(16 16)
bs_arr=(32 12)
num_bs_arr=(8 12)
percent_arr=("0 50 0 0 0 100" "0 50 0 0 0 100")

for(( i=0;i<${#model_name_arr[@]};i++)) do
   bs=${bs_arr[i]};
   num_bs=${num_bs_arr[i]};
   input_len=${input_len_arr[i]};
   output_len=${out_len_arr[i]};
   model_name=${model_name_arr[i]};
   percent=${percent_arr[i]};
   model_path=${model_prefix}/${model_name};
   gpu_log="${gpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}.qdrep";
   cpu_log="${cpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}.txt";
   cpu_log_org="${cpu_log}.org";
   args_str="--model ${model_path} --path _DUMMY_ --pin-weight 0 --percent ${percent} --gpu-batch-size ${bs} --num-gpu-batches ${num_bs}  --prompt-len ${input_len} --gen-len ${out_len}  --cpu";
   cmd="python flex_opt.py $args_str --log-file ${cpu_log_org}";
   echo $cmd;
   $cmd;
   cmd="nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} python flex_opt_prof.py $args_str ";
   echo $cmd;
   $cmd;
   echo "done";
done

