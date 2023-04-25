log_dir=/workspace/data/deep_logs/v1
mkdir -p ${log_dir}
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}

model_prefix="facebook"
model_name_arr=("opt-175b" "opt-175b" "opt-175b" "opt-175b")
input_len_arr=(512 512 512 512)
out_len_arr=(8 8 8 8)
num_gpu_arr=(1 2 4 8)
bs_arr=(1 2 4 8)
num_bs=1
percent=("0 100 0 100 0 100")

for(( i=0;i<${#model_name_arr[@]};i++)) do
   bs=${bs_arr[i]};
   input_len=${input_len_arr[i]};
   out_len=${out_len_arr[i]};
   model_name=${model_name_arr[i]};
   ps=${percent// /:}
   model_path=${model_prefix}/${model_name};
   num_gpu=${num_gpu_arr[i]};
   gpu_log="${gpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}_${ps}_0_${num_gpu}.qdrep";
   cpu_log="${cpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}_${ps}_0_${num_gpu}.txt";
   cpu_log_org="${cpu_log}.org";
   cmd_pre="deepspeed --num_nodes 1 --num_gpus ${num_gpu} --master_port 7778"
   args_str="--model ${model_path} ${off_dir_args} --dummy --cpu-offload --batch-size ${bs}  --prompt-len ${input_len} --gen-len ${out_len} ";
   cmd="${cmd_pre} hf_opt.py $args_str --log-file ${cpu_log_org}";
   echo $cmd;
   $cmd;
   cmd="nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} ${cmd_pre} hf_opt_prof.py $args_str --cpu_log_path ${cpu_log}";
   echo $cmd;
   $cmd;
   echo "done";
done


