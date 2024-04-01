#!/bin/bash
sudo sysctl vm.nr_hugepages=4000
# Get the absolute path of the base directory (two levels up from the script's directory)
readonly BASEDIR=$(readlink -f "$(dirname "$0")")/../../..
# Get the absolute path of the script's directory and set it as the app path
SCRIPT_PATH=$(readlink -f "$(dirname "$0")")/
echo $SCRIPT_PATH
APP_MLC=$SCRIPT_PATH/mlc/Linux/mlc
APP_STREAM=$SCRIPT_PATH/STREAM/stream


# Set the memory type to "cxl" by default
MEMTYPE=cxl

# Set the memory 
MEMSIZE_MB=300000

# Set the directory to store the results
RESULTS_DIR=./results
system=$1
CGROUP_NAME=stretch_cxl

# Set the results directory
res_dir=${RESULTS_DIR}

# Create the results directory if it doesn't exist
mkdir -p "$res_dir"

# Print the page size
getconf -a | grep PAGE_SIZE

# Check whether the memory control group exists and create it if it doesn't
# if [ ! -d "/sys/fs/cgroup/memory/${CGROUP_NAME}" ]; then
#     cgcreate -a "$USER:$USER" -g memory:"${CGROUP_NAME}"
# fi

# Define a usage function
function usage(){
    echo "Usage: $0 [--cxl-check | --cxl-stream-benchmark | --mem0-stream-benchmark | --mem1-stream-benchmark | --cxl-data-latency | --cxl-bandwidth | --mem0-bandwidth | --mem1-bandwidth | --bandwidth-matrix | --latency-matrix | --latency-matrix-all]"
    exit 2
}

# Set the memory set to 0 by default
MEM_SET=0
CMD=''
PORT=9808
MODE=''
# Calculate the memory limit in bytes based on the memory size
# CGROUP_MEM_BYTES=$((MEMSIZE_MB*1024**2))
# Set the memory limit for the control group
# echo "${CGROUP_MEM_BYTES}" > "/sys/fs/cgroup/memory/${CGROUP_NAME}/memory.limit_in_bytes"

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cxl-check)
            # Set the memory type to "cxl" 
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                
                echo "================================================="
                echo "Protowave card CXL memory enumeration by OS "
                echo "================================================="
                lspci | grep CXL
                
            fi
            shift
            ;;
        --cxl-stream-benchmark)
            # Set the memory type to "cxl" 
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                
                echo "================================================="
                echo "Testing HPC operations Bandwidth for CXL "
                echo "================================================="
                numactl --cpunodebind=0  --membind=$MEM_SET $APP_STREAM -a 100000000
                
            fi
            shift
            ;;
        
        --mem0-stream-benchmark)
            # Set the memory type to "cxl" 
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                
                echo "================================================="
                echo "Testing HPC operations Bandwidth for Mem0 "
                echo "================================================="
                MEM_SET=0
                numactl --cpunodebind=0  --membind=$MEM_SET $APP_STREAM -a 100000000
                
            fi
            shift
            ;;

        --mem1-stream-benchmark)
            # Set the memory type to "cxl" 
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                echo "================================================="
                echo "Testing HPC operations Bandwidth for Mem1 "
                echo "================================================="
                MEM_SET=1
                numactl --cpunodebind=1  --membind=$MEM_SET $APP_STREAM -a 100000000
                
            fi
            shift
            ;;
        
        --cxl-data-latency)
            
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                echo "==================================================="
                echo "Testing Data transfer latency from Node 0#CPU to CXL"
                echo "==================================================="
                $APP_MLC --idle_latency -c0 -j2 -b1
                $APP_MLC --idle_latency -c0 -j2 -b1m
                $APP_MLC --idle_latency -c0 -j2 -b16m
                $APP_MLC --idle_latency -c0 -j2 -b64m
                $APP_MLC --idle_latency -c0 -j2 -b1g
                $APP_MLC --idle_latency -c0 -j2 -b16g

            fi
            shift
            ;;
        --cxl-bandwidth)
            # Set the memory type to "cxl" and the memory set to 1
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=cxl
                MEM_SET=2
                echo "================================================="
                echo "Testing bandwidth for Numa Node 0 and CXL Node 2"
                echo "================================================="
                numactl --cpunodebind=0  --membind=$MEM_SET $APP_MLC --max_bandwidth -b200m
                echo "================================================="
                echo "Testing bandwidth for Numa Node 1 and CXL Node 2"
                echo "================================================="
                numactl --cpunodebind=1 --membind=$MEM_SET $APP_MLC  --max_bandwidth -b200m

            fi
            shift
            ;;
        --mem0-bandwidth)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal0
                MEM_SET=0
                echo "================================================="
                echo "Testing bandwidth for Numa Node 0 and Mem Node 0"
                echo "================================================="
                numactl --cpunodebind=0  --membind=$MEM_SET $APP_MLC --max_bandwidth -b200m
                echo "================================================="
                echo "Testing bandwidth for Numa Node 1 and Mem Node 0"
                echo "================================================="
                numactl --cpunodebind=1 --membind=$MEM_SET $APP_MLC  --max_bandwidth -b200m


            fi
            shift
            ;;
         --mem1-bandwidth)
            # Set the memory type to "normal" and the memory set to 0
            if [ $MEM_SET -eq 0 ]; then
                MEMTYPE=normal1
                MEM_SET=1
                echo "================================================="
                echo "Testing bandwidth for Numa Node 0 and Mem Node 1"
                echo "================================================="
                numactl --cpunodebind=0  --membind=$MEM_SET $APP_MLC --max_bandwidth -b200m
                echo "================================================="
                echo "Testing bandwidth for Numa Node 1 and Mem Node 1"
                echo "================================================="
                numactl --cpunodebind=1 --membind=$MEM_SET $APP_MLC  --max_bandwidth -b200m

            fi
            shift
            ;;
        --bandwidth-matrix)
            # Set the memory type to "interleave" and the memory set to 0,2
            if [ $MEM_SET -eq 0 ]; then
                echo "================================================="
                echo "Testing bandwidth for all nodes: Bandwidth Matrix"
                echo "================================================="
                $APP_MLC --bandwidth_matrix -W3 -b200m
            fi
            shift
            ;;
        --latency-matrix)
            # Set the memory type to "interleave" and the memory set to 0,2
            if [ $MEM_SET -eq 0 ]; then
                echo "================================================="
                echo "Testing latency for all nodes: Latency Matrix"
                echo "================================================="
                $APP_MLC --latency_matrix -b200m
            fi
            shift
            ;;
        --latency-matrix-all)
            # Set the memory type to "interleave" and the memory set to 0,2
            if [ $MEM_SET -eq 0 ]; then
                echo "================================================="
                echo "Testing latency for all cpu cores: Latency Matrix"
                echo "================================================="
                $APP_MLC --latency_matrix -a -b200m
            fi
            shift
            ;;
        *)
            # If an invalid option is provided, call the usage function
            usage
            ;;
    esac
done


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
