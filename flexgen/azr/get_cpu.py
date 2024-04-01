import re
import subprocess

# Command to execute
command = "lstopo-no-graphics --no-caches --no-icaches --no-useless-caches --no-smt | grep -v Core"
# Execute the command and capture the output
lstopo_output = subprocess.check_output(command, shell=True, universal_newlines=True)

def get_numa_nodes():
    # Execute numactl -H command
    numctl_output = subprocess.check_output(["numactl", "-H"], universal_newlines=True)
    # Parse the output to find NUMA nodes with no CPUs
    numa_nodes_with_no_cpus = []
    lines = numctl_output.split("\n")
    for line in lines:
        if "node" in line and "cpus:" in line:
            parts = line.split()
            node_id = parts[1]
            cpus = parts[3:]
            if not cpus:
                numa_nodes_with_no_cpus.append(node_id)
    # Print the NUMA nodes with no CPUs
    if numa_nodes_with_no_cpus:
        for node_id in numa_nodes_with_no_cpus:
            print(f"FOUND CXL AS NUMA NODE: {node_id}")
    else:
        print("No NUMA nodes with no CPUs found.")
    return numa_nodes_with_no_cpus

numa_nodes_with_no_cpus = get_numa_nodes()
output = lstopo_output
packages = {}
current_package = None
current_numanode = None

for line in output.split('\n'):
    line = line.strip()
    if line.startswith("Package"):
        current_package = line
        packages[current_package] = []
    elif line.startswith("NUMANode"):
        current_numanode = line
        packages[current_package].append(current_numanode)

for package, numanodes in packages.items():
    for numanode in numanodes:
        # print(numanode)
        for node in numa_nodes_with_no_cpus:
            if f'P#{node}' in numanode:
                print(f"CXL {node} is connected to {package}")
        
