# Call the Python script and capture its output
output=$(python3 get_cpu.py)
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