#!/bin/bash

# Set the path to the Python executable
PYTHON=python # use in Docker

# PYTHON=/home/ansible/miniconda3/envs/ui38/bin/python # use outside docker
# Get the absolute path of the base directory (two levels up from the script's directory)
readonly BASEDIR=$(readlink -f "$(dirname "$0")")/../../..

# Get the absolute path of the script's directory and set it as the app path
SCRIPT_PATH=$(readlink -f "$(dirname "$0")")/

current_user=$(whoami)

# Set the directory to store the results
system=$1


# Call the Python script and capture its output
output=$($PYTHON ../flexgen/get_cpu_cxl.py)
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
numactl --membind=${nodeid} --cpunodebind=${nodeid} $PYTHON serv/main.py
