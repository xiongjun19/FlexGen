#!/bin/bash

# Run the Python script and capture its PID
python ../../project_v4/serv/main.py &
python_pid=$!

# Wait for the Python script to start and obtain its PID
sleep 10

# Find the exact PID of the Python process based on the script name
python_pid=$(pgrep -f "serv/main.py")

# Check if the Python process is running
if [ -n "$python_pid" ]; then
    echo "Python process is running with PID: $python_pid"
else
    echo "Python process is not running"
    exit 1
fi

# Perform some actions...

# Terminate the Python process
kill "$python_pid"
