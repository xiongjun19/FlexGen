#!/bin/bash

# Set the path to the Python executable
PYTHON=/opt/conda/bin/python # use in Docker

# PYTHON=/home/ansible/miniconda3/envs/ui38/bin/python # use outside docker
# Get the absolute path of the base directory (two levels up from the script's directory)
readonly BASEDIR=$(readlink -f "$(dirname "$0")")/../../..

# Get the absolute path of the script's directory and set it as the app path
SCRIPT_PATH=$(readlink -f "$(dirname "$0")")/

# Set the memory type to "cxl" by default
MEMTYPE=cxl

current_user=$(whoami)

# Set the directory to store the results
RESULTS_DIR=./results
system=$1

MODEL='--model=facebook/opt-1.3b'
# Set the results directory
res_dir=${RESULTS_DIR}

# Create the results directory if it doesn't exist
mkdir -p "$res_dir"

# Print the page size
getconf -a | grep PAGE_SIZE


directory="tmp"
 rm -rf tmp
# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Creating $directory directory..."
  mkdir "$directory"
  echo "Directory created successfully."
else
  echo "$directory directory already exists."
  
fi


# Define a usage function
function usage(){
    echo "Usage: $0 [ --cxl-offload | --cxl-offload-sim | --memverge-offload | --normal-offload | --disk-offload | --normal1-offload]"
    exit 2
}

# Set the memory set to 0 by default
MEM_SET=0
CMD=''
PORT=9808
batch_size=24
# sudo service redis stop
# sudo rm -rf message.txt
# sudo chmod +x $SCRIPT_PATH/startmm.sh
# sudo chmod +x $SCRIPT_PATH/stopmm.sh
# echo "Check Memverge status..."

# Following two steps are very important to stop the machine if it is already started.
# The reason to add starmm.sh is to ensure a fail-safe mechanism to gracefully shutdown the mm.
# sudo $SCRIPT_PATH/startmm.sh 
# sudo $SCRIPT_PATH/stopmm.sh


# Call the Python script and capture its output
output=$($PYTHON get_cpu_cxl.py)
# Extract the package number from the output
package_number=$(echo "$output" | grep -o 'Package L#[0-9]*' | awk -F'#' '{print $2}')
# Print the package number
echo "CHECK: CXL connected to CPU Package Number: $package_number"

if [ "$package_number" = "0" ]; then
    nodeid=1
elif [ "$package_number" = "1" ]; then
    nodeid=0
else
    echo "Invalid package number"
    exit 1
fi

# Print the nodeid
echo "Using CPU NODE ID: $nodeid"

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cxl-offload)
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                log_file='OPT-66b-CXL-OUTPUT.log'
                echo "stop" > message.txt
                echo "start cxl" > message.txt
                $PYTHON mem_logger.py online_cxl.csv &
                # Use Cross nodes, if CXL is on Socket0 then select --cpunodebind=1, and vice versa
                numactl --membind=$MEM_SET --cpunodebind=${nodeid} $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir /workspace/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
                
            fi
            shift
            ;;
        --cxl-offload-sim)
            # Set the memory type to "cxl" and the memory set to 1
            CGROUP_NAME="cxl_control_${current_user}"
            # Check whether the memory control group exists and create it if it doesn't
            if [ ! -d "/sys/fs/cgroup/memory/${CGROUP_NAME}" ]; then
                sudo cgcreate -a "$USER:$USER" -g memory:"${CGROUP_NAME}"
            fi
            # Check whether the blkio control group exists and create it if it doesn't
            if [ ! -d "/sys/fs/cgroup/blkio/${CGROUP_NAME}" ]; then
                sudo cgcreate -a "$USER:$USER" -g blkio:"${CGROUP_NAME}"
            fi
            # Set the memory maximum size for the control group
            MEMSIZE_B=80000 #70000->0.842  #90000->1.65 #80000->1.671 # better use above 80000
            # Calculate the memory limit in bytes based on the memory size
            CGROUP_MEM_BYTES=$((MEMSIZE_B*1024**2))

            # Set the memory limit for the control group
            sudo echo "${CGROUP_MEM_BYTES}" > "/sys/fs/cgroup/memory/${CGROUP_NAME}/memory.limit_in_bytes"
            
            # BLKIO NOT EFFECTIVE DONT USE
            # # Set the blkio throttle read and write limits for the control group
            # BLKIO_THROTTLE_BPS=1874848768 
            # # Set the read bps limit
            # echo "8:0 ${BLKIO_THROTTLE_BPS}" | sudo tee "/sys/fs/cgroup/blkio/${CGROUP_NAME}/blkio.throttle.read_bps_device"
            # # Set the write bps limit
            # echo "8:0 ${BLKIO_THROTTLE_BPS}" | sudo tee "/sys/fs/cgroup/blkio/${CGROUP_NAME}/blkio.throttle.write_bps_device"

            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl-sim
                MEM_SET=2
                log_file='OPT-66b-CXL-SIM-OUTPUT.log'
                echo "stop" > message.txt
                echo "start cxl-sim" > message.txt
                $PYTHON mem_logger.py online_cxl-sim.csv &
                sudo cgexec -g memory:${CGROUP_NAME} numactl --interleave=$MEM_SET  $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir /workspace/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
                
            fi
            shift
            ;;
        --memverge-offload)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                
                
                echo "stop" > message.txt
                echo "start memverge" > message.txt
                log_file='OPT-66b-MEMVERGE-OUTPUT.log'
                sudo $SCRIPT_PATH/stopmm.sh
                $PYTHON mem_logger.py online_memverge.csv &
                sudo $SCRIPT_PATH/startmm.sh
                percentage=60 # set to 10 for giving 25GB almost from CXL
                /opt/memverge/bin/mm --config $SCRIPT_PATH/mvmalloc_configs/mvmalloc-${percentage}.yml $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
                
                
            fi
            shift
            ;;
        --memverge-offload-scan)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
            
            # Define an array with the percentages
            percentages=("01" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
            
            # Loop through the array elements
            for percentage in "${percentages[@]}"; do
                # Construct the filenames and log file
                csv_file="online_memverge_${percentage}p_${batch_size}b.csv"
                log_file="online/OPT-66b-MEMVERGE-OUTPUT-${percentage}.log"

                # Run the commands
                echo "stop" > message.txt
                echo "start memverge" > message.txt
                $PYTHON mem_logger.py "$csv_file" &
                sudo $SCRIPT_PATH/stopmm.sh
                sudo $SCRIPT_PATH/startmm.sh
                echo "[INFO] Starting Local Memory Percentage as ${percentage} % ..."
                sudo /opt/memverge/bin/mm --config $SCRIPT_PATH/mvmalloc_configs/mvmalloc-${percentage}.yml $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size "${batch_size}" --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file "${log_file}"
                echo "stop" > message.txt
                sudo $SCRIPT_PATH/stopmm.sh
                
                sleep 60
            done
            
            fi
            shift
            ;;
        --normal-offload)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal
                MEM_SET=0
                echo "stop" > message.txt
                echo "start mem" > message.txt
                $PYTHON mem_logger.py online_mem.csv &
                log_file='OPT-66b-MEM-OUTPUT.log'
                numactl --interleave=$MEM_SET $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        --all-offload)
            # Set the memory type to "all" to use about 26% of CXL memory and rest others by interleaving all
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal+cxl
                MEM_SET=0
                echo "stop" > message.txt
                echo "start all" > message.txt
                $PYTHON mem_logger.py online_all.csv &
                log_file='OPT-66b-ALL-OUTPUT.log'
                numactl --interleave=0,1,2 $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        --normal1-offload)
            # Set the memory type to "normal" and the memory set to 1
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal1
                MEM_SET=1
                echo "stop" > message.txt
                echo "start mem1" > message.txt
                $PYTHON mem_logger.py online_mem1.csv &
                log_file='OPT-66b-MEM1-OUTPUT.log'
                numactl --interleave=$MEM_SET $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
                echo "stop" > message.txt
            fi
            shift
            ;;
        
        --disk-offload)
            # Set the memory type 
            if [ $MEM_SET -eq 0 ]; then
                    # Set the memory type to "cxl" and the memory set to 1
                CGROUP_NAME="disk_control_${current_user}"
                # Check whether the memory control group exists and create it if it doesn't
                if [ ! -d "/sys/fs/cgroup/memory/${CGROUP_NAME}" ]; then
                    cgcreate -a "$USER:$USER" -g memory:"${CGROUP_NAME}"
                fi
                # Check whether the blkio control group exists and create it if it doesn't
                if [ ! -d "/sys/fs/cgroup/blkio/${CGROUP_NAME}" ]; then
                    cgcreate -a "$USER:$USER" -g blkio:"${CGROUP_NAME}"
                fi
                # This limit will make sure that Disk will not use more than 20GB of memory and disk-offloading will not use any cached data
                MEMSIZE_B=20000
                # Calculate the memory limit in bytes based on the memory size
                CGROUP_MEM_BYTES=$((MEMSIZE_B*1024**2))
                # Set the memory limit for the control group
                echo "${CGROUP_MEM_BYTES}" > "/sys/fs/cgroup/memory/${CGROUP_NAME}/memory.limit_in_bytes"
                MEMTYPE=disk
                CMD='--disk-offload'
                MEM_SET=0
                echo "stop" > message.txt
                echo "start disk" > message.txt
                $PYTHON mem_logger.py online_disk.csv &
                log_file='OPT-66b-DISK-OUTPUT.log'
                cgexec -g memory:${CGROUP_NAME} $PYTHON flex_opt.py --model facebook/opt-66b --offload-dir /tmp/data/flex_offload_dir --path _DUMMY_ --percent 0 0 0 0 0 100 --gpu-batch-size ${batch_size} --num-gpu-batches 4 --prompt-len 512 --gen-len 8 --compress-weight --compress-cache --log-file ${log_file}
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
