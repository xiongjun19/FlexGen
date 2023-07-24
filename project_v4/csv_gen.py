import csv
import random
import time
import os
# Constant values for non-percentage keys
PRODUCT = "NVIDIA A10"
PCI_INFO = "3Gen-16xlanes"
GPUMEM_USED_MB = 573.4375
GPUMEM_TOTAL_MB = 15360.0
os.makedirs('online',exist_ok=True)
os.makedirs('decode_throughput',exist_ok=True)
# Function to generate a random value between 0 and 100 for percentage keys
def generate_random_percentage(num):
    return round(random.uniform(0, num), 1)
# Infinite loop to continuously generate and append data
while True:
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    # Print the header to the CSV file
    print("TIME,CPU%,MEM%,GPU%,GPUMEM%,CXLMEM%,PCI_TX_MBps,PCI_RX_MBps,PRODUCT,PCI_INFO,GPUMEM_USED_MB,GPUMEM_TOTAL_MB", file=open('online/online_cxl.csv', 'w'))
    # Generate and append data for 1400 iterations (you may set it to low value for faster debugging)
    for _ in range(20):
        text = f"{current_time},{generate_random_percentage(100)},{generate_random_percentage(100)},{generate_random_percentage(100)},{generate_random_percentage(100)},{generate_random_percentage(100)},0.0,0.0,{PRODUCT},{PCI_INFO},{GPUMEM_USED_MB},{GPUMEM_TOTAL_MB}"
        print(text, file=open('online/online_cxl.csv', 'a'))
        print(text)
        time.sleep(1)
    # Generate new decode throughput once run is finished
    print(generate_random_percentage(2), file=open('decode_throughput/cxl.log', 'w'))
