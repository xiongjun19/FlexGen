#!/bin/bash


# Set the number of iterations (use -1 for infinite loop)
COUNT=30

# Function to run the rundk-large.sh script
run_script() {
    ./rundk-large-inside-docker.sh --cxl-offload
}

# Infinite while loop when COUNT is -1, or run COUNT times
if [ "$COUNT" -eq -1 ]; then
    echo "Running $RUN_SCRIPT script infinitely..."
    while true; do
        run_script
        sleep 5
    done
else
    
    for ((i = 1; i <= COUNT; i++)); do
        echo "Running $RUN_SCRIPT script $i th time..."
        run_script
        sleep 20
    done
fi
