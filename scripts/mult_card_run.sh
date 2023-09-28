#!/bin/bash

N_GPUS=4
N_CORES_PER_GPU=12


pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x
log_dir=/workspace/data/flex_logs_mult_cards
mkdir -p ${log_dir}
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}


model_prefix="facebook"
model_name_arr=("opt-66b")
input_len_arr=(512)
out_len_arr=(8)
bs_arr=(16)
num_bs_arr=(2)
percent_arr=("0 100 0 100 100 0")


for(( i=0;i<${#model_name_arr[@]};i++)) do
   bs=${bs_arr[i]};
   num_bs=${num_bs_arr[i]};
   input_len=${input_len_arr[i]};
   out_len=${out_len_arr[i]};
   model_name=${model_name_arr[i]};
   percent=${percent_arr[i]};
   model_path=${model_prefix}/${model_name};
   gpu_log="${gpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}.qdrep";
   cpu_log="${cpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}.txt";
   cpu_log_org="${cpu_log}.org";
   mpirun --allow-run-as-root \
     --mca btl_tcp_if_exclude lo,docker0 \
     --mca oob_tcp_if_exclude lo,docker0 \
     --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe \
     --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
     python dist_flex_opt.py \
       --head-ip 'localhost' \
       --port 7777 \
       --use-mpi \
       --model ${model_path} \
       --gpu-batch-size ${bs} \
       --percent ${percent} \
       --comm-device cpu \
       --path _DUMMY_ \
       --pin-weight 1 \
       --num-gpu-batches ${num_bs}  --prompt-len ${input_len} --gen-len ${out_len} \
       --log-file ${cpu_log_org};
   echo "done";
done


