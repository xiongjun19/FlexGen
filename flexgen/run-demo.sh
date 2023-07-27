#!/bin/bash


# Set the number of iterations (use -1 for infinite loop)
COUNT=10

# Function to run the rundk-large.sh script
run_script() {
    ./rundk-large.sh --cxl-offload
}

# Infinite while loop when COUNT is -1, or run COUNT times
if [ "$COUNT" -eq -1 ]; then
    echo "Running $RUN_SCRIPT script infinitely..."
    while true; do
        run_script
        sleep 2
    done
else
    echo "Running $RUN_SCRIPT script $COUNT times..."
    for ((i = 1; i <= COUNT; i++)); do
        run_script
        sleep 2
    done
fi
