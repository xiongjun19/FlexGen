import subprocess
import csv
import time
import os
import gc

data_sizes = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024]


output_file = 'output.csv'

# Write the CSV header
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['data_size_KB','device_id','device_type' ,'mem_type', 'memory_to_device_GB/s', 'device_to_memory_GB/s'])

mem_type = ['mem0','mem1','cxl'] # only use if cxl is there as numa node 2
# mem_type = ['cxl'] # use only if no cxl is there

device_ids =[f"{i}" for i in range(1)]

for device_id in device_ids:
    gc.collect()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
        print(f"Testing for Device ID: {device_id}")
        for idx,mem in enumerate(mem_type):
            # idx=2
            # Run the command for each data size
            for size in data_sizes:
                command = f"numactl --membind={idx} ./profile_test --mem-type={mem} --data-size={size}"
                result = subprocess.run(command, capture_output=True, text=True, shell=True)
                output = result.stdout.strip()
                
                print(output)
                # Parse the relevant values from the command output
                lines = output.split('\n')
                transfer1_bandwidth = lines[-2].split(':')[1].strip()
                transfer2_bandwidth = lines[-1].split(':')[1].strip()
                device_type = lines[2].strip('Device: ')
                # Append the data to the CSV file
                with open(output_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([size/1024,device_id,device_type,mem, transfer1_bandwidth, transfer2_bandwidth])
                time.sleep(1)
    except:
        pass
print("Output saved to", output_file)
