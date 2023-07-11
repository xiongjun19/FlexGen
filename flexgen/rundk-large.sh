#!/bin/bash

# Set the path to the Python executable
PYTHON=/opt/conda/bin/python # use in Docker
# PYTHON=/home/miniconda3/envs/ui38/bin/python # use outside docker
# Get the absolute path of the base directory (two levels up from the script's directory)
readonly BASEDIR=$(readlink -f "$(dirname "$0")")/../../..

# Get the absolute path of the script's directory and set it as the app path
SCRIPT_PATH=$(readlink -f "$(dirname "$0")")/


# Set the memory type to "cxl" by default
MEMTYPE=cxl

# Set the memory maximum size for the control group
MEMSIZE_MB=300000

# Set the directory to store the results
RESULTS_DIR=./results
system=$1
CGROUP_NAME=stretch_cxl
MODEL='--model=facebook/opt-1.3b'
# Set the results directory
res_dir=${RESULTS_DIR}

# Create the results directory if it doesn't exist
mkdir -p "$res_dir"

# Print the page size
getconf -a | grep PAGE_SIZE


directory="tmp"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Creating $directory directory..."
  mkdir "$directory"
  echo "Directory created successfully."
else
  echo "$directory directory already exists."
fi

# Check whether the memory control group exists and create it if it doesn't
# if [ ! -d "/sys/fs/cgroup/memory/${CGROUP_NAME}" ]; then
#     cgcreate -a "$USER:$USER" -g memory:"${CGROUP_NAME}"
# fi

# Define a usage function
function usage(){
    echo "Usage: $0 [ --cxl-offload | --normal-offload | --disk-offload | --normal1-offload]"
    exit 2
}

# Set the memory set to 0 by default
MEM_SET=0
CMD=''
PORT=9808
batch_size=64

# # Calculate the memory limit in bytes based on the memory size
# CGROUP_MEM_BYTES=$((MEMSIZE_MB*1024**2))

# # Set the memory limit for the control group
# echo "${CGROUP_MEM_BYTES}" > "/sys/fs/cgroup/memory/${CGROUP_NAME}/memory.limit_in_bytes"

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cxl-offload)
            # Set the memory type to "cxl" and the memory set to 1
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                log_file='OPT-66b-CXL-OUTPUT.log'
                echo "stop" > message.txt
                echo "start" > message.txt
                $PYTHON mem_logger.py online_cxl.csv &
                numactl --interleave=$MEM_SET $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir /workspace/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        --memverge-offload)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                echo "stop" > message.txt
                echo "start" > message.txt
                $PYTHON mem_logger.py online_memverge.csv &
                log_file='OPT-66b-MEMVERGE-OUTPUT.log'
                mm --config /home/ahussain/azr/workspace/FlexGen/flexgen/mvmalloc.yml $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        --normal-offload)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal
                MEM_SET=0
                echo "stop" > message.txt
                echo "start" > message.txt
                $PYTHON mem_logger.py online_mem.csv &
                log_file='OPT-66b-NORMAL-OUTPUT.log'
                numactl --interleave=$MEM_SET $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        
        --normal1-offload)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal
                MEM_SET=1
                echo "stop" > message.txt
                echo "start" > message.txt
                $PYTHON mem_logger.py online_mem1.csv &
                log_file='OPT-66b-NORMAL1-OUTPUT.log'
                numactl --interleave=$MEM_SET $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        
        --disk-offload)
            # Set the memory type to "cxl" and the memory set to 2
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=disk
                CMD='--disk-offload'
                MEM_SET=0
                echo "stop" > message.txt
                echo "start" > message.txt
                $PYTHON mem_logger.py online_disk.csv &
                log_file='OPT-66b-DISK-OUTPUT.log'
                numactl --interleave=$MEM_SET $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 0 0 0 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;

        *)
            # If an invalid option is provided, call the usage function
            usage
            ;;
    esac
done



# Execute the app with the specified options
# cgexec -g memory:${CGROUP_NAME} numactl --physcpubind=+0-15,16-34 --interleave=$MEM_SET $PYTHON $APP --mem-type=$MEMTYPE $CMD $MODEL #PORT=9808

# Get the return code of the app
ret=$?

# Unset LD_PRELOAD
unset LD_PRELOAD

# Remove the file specified by the FILE variable if it is a character device
if [ -c "$FILE" ]; then
    rm -rf "$FILE"
fi

# Print a message indicating whether the app passed or failed
echo
if [ $ret -eq 0 ]; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Exit the script with a successful status code
exit
