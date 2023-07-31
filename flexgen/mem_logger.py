import psutil
import time
import csv
import time
import numpy as np
from tabulate import tabulate
import random
import numpy as np
import os, sys
import subprocess
import re
from pynvml.smi import nvidia_smi
import signal


def signal_handler(sig, frame):
    print("Ctrl+C pressed. Exiting...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
nvsmi = nvidia_smi.getInstance()
nvsmi.DeviceQuery()['gpu'][0]['pci']
nvsmi.DeviceQuery()['gpu'][0].keys()
try:
    name =sys.argv[1]
except Exception as e:
    print(f'logging data to online/test.csv because {e}')
    name='test.csv'

os.makedirs('online',exist_ok=True)
def extract_pmemory_usage():
    command = "mvmcli show-usage"
    output = subprocess.check_output(command, shell=True, text=True)
    
    # Regular expressions to extract the required information
    dram_total_pattern = r"Managed DRAM\(MiB\): Total (\d+\.\d+)"
    dram_used_pattern = r"DRAM\(MiB\): Total \d+\.\d+, Used (\d+\.\d+)"
    pmem_total_pattern = r"Managed PMEM\(MiB\): Total (\d+\.\d+)"
    pmem_used_pattern = r"PMEM\(MiB\): Total \d+\.\d+, Used (\d+\.\d+)"
    
    # Extract total DRAM and used DRAM values
    dram_total_match = re.search(dram_total_pattern, output)
    dram_used_match = re.search(dram_used_pattern, output)
    
    # Extract total PMEM and used PMEM values
    pmem_total_match = re.search(pmem_total_pattern, output)
    pmem_used_match = re.search(pmem_used_pattern, output)
    
    if dram_total_match and dram_used_match and pmem_total_match and pmem_used_match:
        dram_total = float(dram_total_match.group(1))
        dram_used = float(dram_used_match.group(1))
        pmem_total = float(pmem_total_match.group(1))
        pmem_used = float(pmem_used_match.group(1))
        return dram_total, dram_used, pmem_total, pmem_used
    
    return None
def get_max_memory_info(node_id):
    with open('/proc/zoneinfo', 'r') as f:
        content = f.read()
    max_memory = {}
    try:
        zones = content.split(f'Node {node_id}, zone')[1:]  
        
        for zone in zones:
            name = zone.split('\n')[0].strip()
            if name in ('DMA', 'DMA32', 'Normal'):  # select the four zones
                managed_pages = int(zone.split('managed')[1].split('\n')[0].strip())
                print(f"{name}: {managed_pages} pages")
                max_memory[name] = managed_pages*4096/1024**2
        return max_memory
    except:
        return 0

CONSTANTS = np.array([4*1024,   8*1024, 16*1024,    32*1024,    64*1024,    128*1024,   256*1024,   512*1024,   1 *1024**2, 2 *1024**2, 4*1024**2])
headers = ['       TIME', 'USED  \n CXL(MB)', 'FREE  \n CXL(MB)','USED   \n NORMAL-0(MB)','FREE   \nNORMAL-0(MB)','USED   \n NORMAL-1(MB)','FREE   \nNORMAL-1(MB)']

def get_memory_info():
    with open('/proc/buddyinfo') as f:
        lines = f.readlines()
    
    memory_info = {}
    for line in lines:
        
        if line.startswith('Node'):
            
            _, id, _,zone, *counts = line.split()
            id = int(id[0:-1])
            counts = np.array([int(x) for x in counts])
            counts = np.multiply(counts,CONSTANTS)/1024**2
            memory_info[(id,zone)] = np.sum(counts)
    
    return memory_info

    

data = []
max_memory = {}
max_memory[0] = get_max_memory_info(0)
max_memory[1] = get_max_memory_info(1)
max_memory[2] = get_max_memory_info(2)

log_csv = False
prev_used = {}
prev_used['exmem'] = None
prev_used['normal'] = None
prev_used['normal1'] = None
duration_ms=1000
bandwidth = {}
bandwidth['exmem'] = 0
bandwidth['normal'] = 0
bandwidth['normal1'] = 0

def update():
    global prev_used, bandwidth
    while True:
        memory_info = get_memory_info()
        dma_free = memory_info[(0,'DMA')]
        dma32_free = memory_info[(0,'DMA32')]
        normal_free = memory_info[(0,'Normal')]

        normal1_free = memory_info[(1,'Normal')]
        used_normal1 = max_memory[1]['Normal'] -  normal1_free

        if (2,'Normal') in memory_info:
            exmem_free = memory_info[(2,'Normal')]
            used_exmem = max_memory[2]['Normal'] -  exmem_free
        else:
            exmem_free = 0
            used_exmem = 0

        used_normal = max_memory[0]['Normal'] - normal_free
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        sample = [current_time, used_exmem, exmem_free, used_normal, normal_free,used_normal1, normal1_free]
        data.append(sample)
        
        
        if len(data) > 8:
            data.pop(0)
        print('\033c') # clear the console
        print(tabulate(data, headers=headers))
        if log_csv:
            with open('memory_info.csv', mode='a') as file:
                writer = csv.writer(file)
                writer.writerow(sample)

        time.sleep(1)





def get_gpu_info(info):
    gpu = info['utilization']['gpu_util']
    gpumem = info['utilization']['memory_util']
    unit =  info['utilization']['unit']
    pci_tx = info['pci']['tx_util']/1024
    pci_rx = info['pci']['rx_util']/1024
    pci_x_unit = info['pci']['tx_util_unit']
    gpumem_used = info['fb_memory_usage']['used']
    gpumem_total = info['fb_memory_usage']['total']
    gpumem_unit = info['fb_memory_usage']['unit']
    product =  info['product_name']
    perfomance_state = info['performance_state']
    gpu_temperature = info['temperature']['gpu_temp']
    temperature_unit = info['temperature']['unit']
    product_gen =  f"{info['pci']['pci_gpu_link_info']['pcie_gen']['current_link_gen']}Gen-{info['pci']['pci_gpu_link_info']['link_widths']['current_link_width']}lanes"
    
    if unit == '%' and pci_x_unit =='KB/s' and gpumem_unit=='MiB' and temperature_unit=='C':
        return gpu,gpumem, pci_tx,pci_rx, product,product_gen, gpumem_used, gpumem_total, perfomance_state,gpu_temperature
    else:
        assert 0 , "units check please"
def get_traces(idx,MODE = 'CXL'):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    gpu_percent,gpumem_percent,pci_tx,pci_rx,product,product_gen, gpumem_used, gpumem_total,performance_state,gpu_temperature = get_gpu_info(info[idx])
    memory_info = get_memory_info()
    pmem_used= 0
    if MODE =='MEMVERGE':
        try:
            memory_usage = extract_pmemory_usage()
            if memory_usage:
                _, _, pmem_total, pmem_used = memory_usage
        except Exception as e:
            print('Warning mvmcli can be ignored if not using Memverge',e)
            # import pdb;pdb.set_trace()
            pass
        
    dma_free = memory_info[(0,'DMA')]
    dma32_free = memory_info[(0,'DMA32')]
    normal_free = memory_info[(0,'Normal')]

    normal1_free = memory_info[(1,'Normal')]
    used_normal1 = max_memory[1]['Normal'] -  normal1_free

    if (2,'Normal') in memory_info:
        exmem_free = memory_info[(2,'Normal')]
        used_exmem = max_memory[2]['Normal'] -  exmem_free
    else:
        exmem_free = 0
        used_exmem = 0

    used_normal = max_memory[0]['Normal'] - normal_free
    if (used_exmem+exmem_free) ==0:
        cxl_percent = 0
    else:
        cxl_percent = 100*used_exmem/(used_exmem+exmem_free)
    if MODE=='DISK':
        MODE="DISK'"
    string = f"{current_time},{idx},{cpu_percent},{ram_percent},{gpu_percent},{gpumem_percent},{cxl_percent},{pci_tx},{pci_rx},{product},{product_gen},{gpumem_used},{gpumem_total},{pmem_used},{performance_state},{gpu_temperature},{MODE}"
    print(string,file=open(f'online/{name}-gpu-{idx}.csv', 'a'))
    print(string)

def get_mode_info(text):
    mode_list = ['cxl-sim','cxl','disk','memverge','mem','mem1']
    for mode in mode_list:
        if mode in text:
            return mode
        
    
        

info  = nvsmi.DeviceQuery()['gpu']
my_gpus_ids_to_monitor = [0]
for idx in my_gpus_ids_to_monitor:
    print(f"TIME,GPU_ID,CPU%,MEM%,GPU%,GPUMEM%,CXLMEM%,PCI_TX_MBps,PCI_RX_MBps,PRODUCT,PCI_INFO,GPUMEM_USED_MB,GPUMEM_TOTAL_MB,CXL_PMEM_USED_MB,PERFORMANCE_STATE,GPU_TEMPERATURE_C,MODE",file=open(f'online/{name}-gpu-{idx}.csv', 'w'))

sample_interval = 0.99 # seconds

while True:
    info  = nvsmi.DeviceQuery()['gpu']
    if os.path.exists('message.txt'):
        with open('message.txt', 'r') as file:
            message = file.read().strip()
        if message == 'stop' or ('stop' in message):
            break
    MODE = get_mode_info(message)
        
    for idx in my_gpus_ids_to_monitor:
        get_traces(idx,MODE.upper())
        
    time.sleep(sample_interval)
print('Memlogger Stopped!!!')
os.remove('message.txt')
sys.exit()

    
