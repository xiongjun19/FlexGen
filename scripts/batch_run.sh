log_dir=/workspace/data/flex_logs
mkdir -p ${log_dir}
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}

model_prefix="facebook"
model_name_arr=("opt-6.7b" "opt-66b" "opt-175b")
input_len_arr=(512 1024)
# input_len_arr=(512)
# out_len_arr=(128 256 512 1024)
out_len_arr=(4 8 16)
bs=1

for(( i=0;i<${#model_name_arr[@]};i++)) do
   for(( j=0;j<${#input_len_arr[@]};j++)) do
     for(( k=0;k<${#out_len_arr[@]};k++)) do
         model_name=${model_name_arr[i]};
         model_path=${model_prefix}/${model_name};
         input_len=${input_len_arr[j]};
	 out_len=${out_len_arr[k]};
	 gpu_log="${gpu_logs}/${model_name}_${input_len}_${out_len}.qdrep";
         cpu_log="${cpu_logs}/${model_name}_${input_len}_${out_len}.txt";
	 cmd="python flex_opt.py --model ${model_path} --path _DUMMY_  --percent 0 100 100 0 100 0 --gpu-batch-size ${bs} --num-gpu-batches 1  --prompt-len ${input_len} --gen-len ${out_len} ";
	 echo $cmd; 
         $cmd;	
	 cmd="nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} python flex_opt_prof.py --model ${model_path} --path _DUMMY_  --percent 0 100 100 0 100 0 --gpu-batch-size ${bs} --num-gpu-batches 1  --prompt-len ${input_len} --gen-len ${out_len} --cpu_log_path ${cpu_log} ";
	 echo $cmd; 
         $cmd;	
	 echo "done";
      done
    done
 done


