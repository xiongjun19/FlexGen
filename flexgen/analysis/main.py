import re
import csv

time_pattern = re.compile(r'TIME:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
i_pattern = re.compile(r'inner most i,j,k: (-?\d+),(-?\d+),(-?\d+)')
data_list = []
with open('mem-opt-66b.log', 'r') as file:
    lines = file.readlines()
for i in range(len(lines)):
    if "TIME:" in lines[i] and "inner most i,j,k:" in lines[i+1]:
        time_match = time_pattern.search(lines[i])
        i_match = i_pattern.search(lines[i+1])
        if time_match and i_match:
            current_time = time_match.group(1)
            i_value = int(i_match.group(1))
            j_value = int(i_match.group(2))
            k_value = int(i_match.group(3))
            data_list.append([current_time, i_value, j_value, k_value])
with open('output.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['TIME', 'i', 'j', 'k'])
    csv_writer.writerows(data_list)
print("Data has been successfully extracted")
