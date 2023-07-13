import csv, os
os.system('python main_gpu_gds.py > gds.log')
input_file = "gds.log"
output_file = "output_gds.csv"

# Open the input file for reading
with open(input_file, "r") as file:
    lines = file.readlines()

# Extract data and metrics from the lines
data = []
read_throughput, write_throughput =-1,-1
size=-1
bw={'rd':-1,'wr':-1}
for line in lines:
    if "DataSetSize" in line:
        values = line.split()
        # import pdb;pdb.set_trace()
        size = values[7]
        if 'WRITE' in values:
            write_throughput = values[11]
            bw['wr'] = write_throughput
        if 'READ' in values:
            read_throughput = values[11]
            bw['rd'] = read_throughput
    if (bw['rd'] != -1) and (bw['wr'] != -1):
        data.append([size, bw['rd'], bw['wr']])
        bw['rd'], bw['wr'] =-1,-1
# Write data to output CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["datasize", "read_throughput_GB/s", "write_throughput_GB/s"])
    writer.writerows(data)

print(f"Output saved to {output_file}")
