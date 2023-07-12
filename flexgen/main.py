# this script runs the resource view in browser: please run it via bash run_viewer.sh
import csv
import time
import numpy as np
from tabulate import tabulate
from bokeh.models import HoverTool
from bokeh.plotting import figure, curdoc
from bokeh.driving import linear
import random
import numpy as np
from bokeh.models import Legend
from bokeh.layouts import gridplot


import subprocess
import re
import psutil
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
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

def get_gpu_info(idx):
        
    info  = nvsmi.DeviceQuery()['gpu'][idx]
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
    gpu_temperature = info['temperature']['gpu_temp']
    product_gen =  f"{info['pci']['pci_gpu_link_info']['pcie_gen']['current_link_gen']}Gen-{info['pci']['pci_gpu_link_info']['link_widths']['current_link_width']}lanes"
    temperature_unit = info['temperature']['unit']
    
    if unit == '%' and pci_x_unit =='KB/s' and gpumem_unit=='MiB' and temperature_unit=='C':
        return gpu,gpumem, pci_tx,pci_rx, product,product_gen, gpumem_used, gpumem_total,gpu_temperature
    else:
        assert 0 , "unit check please"



def get_max_memory_info(node_id):
    with open('/proc/zoneinfo', 'r') as f:
        content = f.read()
    max_memory = {}
    zones = content.split(f'Node {node_id}, zone')[1:]  
    
    for zone in zones:
        name = zone.split('\n')[0].strip()
        if name in ('DMA', 'DMA32', 'Normal'):  # select the four zones
            managed_pages = int(zone.split('managed')[1].split('\n')[0].strip())
            print(f"{name}: {managed_pages} pages")
            max_memory[name] = managed_pages*4096/1024**2
    return max_memory

CONSTANTS = np.array([4*1024,   8*1024, 16*1024,    32*1024,    64*1024,    128*1024,   256*1024,   512*1024,   1 *1024**2, 2 *1024**2, 4*1024**2])
headers = ['       TIME', 'USED  \n CXL(MB)', 'FREE  \n CXL(MB)','USED   \n NORMAL-0(MB)','FREE   \nNORMAL-0(MB)','USED   \n NORMAL-1(MB)','FREE   \nNORMAL-1(MB)']
with open('memory_info.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
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
def write_to_csv():
    data = []
    max_memory = {}
    max_memory[0] = get_max_memory_info(0)
    max_memory[1] = get_max_memory_info(1)
    max_memory[2] = get_max_memory_info(2)
    
    log_csv = False
    
    while True:
        
        memory_info = get_memory_info()
        dma_free = memory_info[(0,'DMA')]
        dma32_free = memory_info[(0,'DMA32')]
        normal_free = memory_info[(0,'Normal')]

        normal1_free = memory_info[(1,'Normal')]
        used_normal1 = max_memory[1]['Normal'] -  normal1_free

        exmem_free = memory_info[(2,'Normal')]
        used_exmem = max_memory[2]['Normal'] -  exmem_free

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

@linear()
def update(step):
    
    memory_info = get_memory_info()
    pmem_used= 0
    try:
        memory_usage = extract_pmemory_usage()
        if memory_usage:
            _, _, pmem_total, pmem_used = memory_usage
    except:
        pass
    
    
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    gpu_idx = 0
    gpu_percent,gpumem_percent,pci_tx,pci_rx,product,product_gen, gpumem_used, gpumem_total,gpu_temperature = get_gpu_info(gpu_idx)
    
    
    
    dma_free = memory_info[(0,'DMA')]
    dma32_free = memory_info[(0,'DMA32')]
    normal_free = memory_info[(0,'Normal')]

    normal1_free = memory_info[(1,'Normal')]
    used_normal1 = max_memory[1]['Normal'] -  normal1_free

    exmem_free = memory_info[(2,'Normal')]
    used_exmem = max_memory[2]['Normal'] -  exmem_free

    used_normal = max_memory[0]['Normal'] - normal_free
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    sample = [current_time, used_exmem, exmem_free, used_normal, normal_free,used_normal1, normal1_free]
    data.append(sample)
    
    # Memory stats
    ds0.data['x'].append(step)
    ds0.data['y'].append(used_exmem)
    ds1.data['x'].append(step)
    ds1.data['y'].append(used_normal)
    ds2.data['x'].append(step)
    ds2.data['y'].append(used_normal1)  
    ds3.data['x'].append(step)
    ds3.data['y'].append(pmem_used)  
    ds4.data['x'].append(step)
    ds4.data['y'].append(gpumem_used)  
    
    ds0.trigger('data', ds0.data, ds0.data)
    ds1.trigger('data', ds1.data, ds1.data)
    ds2.trigger('data', ds2.data, ds2.data)
    ds3.trigger('data', ds3.data, ds3.data)
    ds4.trigger('data', ds4.data, ds4.data)
    
    ## scatter
    dds0.data['x'].append(step)
    dds0.data['y'].append(used_exmem)
    dds1.data['x'].append(step)
    dds1.data['y'].append(used_normal)
    dds2.data['x'].append(step)
    dds2.data['y'].append(used_normal1)  
    dds3.data['x'].append(step)
    dds3.data['y'].append(pmem_used)  
    dds4.data['x'].append(step)
    dds4.data['y'].append(gpumem_used) 
    
    dds0.trigger('data', dds0.data, dds0.data)
    dds1.trigger('data', dds1.data, dds1.data)
    dds2.trigger('data', dds2.data, dds2.data)
    dds3.trigger('data', dds3.data, dds3.data)
    dds4.trigger('data', dds4.data, dds4.data)
    
    # Gpu/Cpu stats
    ds02.data['x'].append(step)
    ds02.data['y'].append(gpu_percent)
    ds12.data['x'].append(step)
    ds12.data['y'].append(gpumem_percent)
    ds22.data['x'].append(step)
    ds22.data['y'].append(cpu_percent)  
    ds32.data['x'].append(step)
    ds32.data['y'].append(ram_percent)  
    ds42.data['x'].append(step)
    ds42.data['y'].append(gpu_temperature)  
    
    ds02.trigger('data', ds02.data, ds02.data)
    ds12.trigger('data', ds12.data, ds12.data)
    ds22.trigger('data', ds22.data, ds22.data)
    ds32.trigger('data', ds32.data, ds32.data)
    ds42.trigger('data', ds42.data, ds42.data)
    ds42.trigger('data', ds42.data, ds42.data)
    
    #scatter
    
    dds02.data['x'].append(step)
    dds02.data['y'].append(gpu_percent)
    dds12.data['x'].append(step)
    dds12.data['y'].append(gpumem_percent)
    dds22.data['x'].append(step)
    dds22.data['y'].append(cpu_percent)  
    dds32.data['x'].append(step)
    dds32.data['y'].append(ram_percent)  
    dds42.data['x'].append(step)
    dds42.data['y'].append(gpu_temperature)  
    
    dds02.trigger('data', dds02.data, dds02.data)
    dds12.trigger('data', dds12.data, dds12.data)
    dds22.trigger('data', dds22.data, dds22.data)
    dds32.trigger('data', dds32.data, dds32.data)
    dds42.trigger('data', dds42.data, dds42.data)
    
    if len(data) > 8:
        data.pop(0)
    # print('\033c') # clear the console
    # print(tabulate(data, headers=headers))
    if log_csv:
        with open('memory_info.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(sample)


def get_p(tags,title, YLABEL, my_colors):
    

    color0,color1,color2,color3,color4 = my_colors[0],my_colors[1],my_colors[2],my_colors[3],my_colors[4]

    p = figure(width=400, height=280)
    p.title.text = title
    p.x_range.follow="end"
    p.x_range.follow_interval = 100
    p.x_range.range_padding=0

    # line_dash="dotdash", ine_dash="dotted"
    r0 = p.line([], [], color=color0, line_width=2,alpha=1,  muted_alpha=0.2, legend_label=tags[0])
    r1 = p.line([], [], color=color1, line_width=2,alpha=1,  muted_alpha=0.2, legend_label=tags[1])
    r2 = p.line([], [], color=color2, line_width=2,alpha=1,  muted_alpha=0.2, legend_label=tags[2])
    r3 = p.line([], [], color=color3, line_width=2,alpha=1,  muted_alpha=0.2, legend_label=tags[3])
    r4 = p.line([], [], color=color4, line_width=2,alpha=1,  muted_alpha=0.2, legend_label=tags[4])

    
    rr0 = p.scatter([], [], color=color0, line_width=1,alpha=.3,  muted_alpha=0.2, legend_label=tags[0])
    rr1 = p.scatter([], [], color=color1, line_width=1,alpha=.3,  muted_alpha=0.2, legend_label=tags[1])
    rr2 = p.scatter([], [], color=color2, line_width=1,alpha=.3,  muted_alpha=0.2, legend_label=tags[2])
    rr3 = p.scatter([], [], color=color3, line_width=1,alpha=.3,  muted_alpha=0.2, legend_label=tags[3])
    rr4 = p.scatter([], [], color=color4, line_width=1,alpha=.3,  muted_alpha=0.2, legend_label=tags[4])

    
    ds0 = r0.data_source
    ds1 = r1.data_source
    ds2 = r2.data_source
    ds3 = r3.data_source
    ds4 = r4.data_source
    
    dds0 = rr0.data_source
    dds1 = rr1.data_source
    dds2 = rr2.data_source
    dds3 = rr3.data_source
    dds4 = rr4.data_source
    
    # add a hover tool for CXL
    hover1 = HoverTool(renderers=[r0], tooltips=[(tags[0], '@y{0.00}')], mode='vline',attachment='left')
    p.add_tools(hover1)

    # add a hover tool for Normal
    hover2 = HoverTool(renderers=[r1], tooltips=[(tags[1], '@y{0.00}')], mode='vline',attachment='right')
    p.add_tools(hover2)

    # add a hover tool for Normal
    hover3 = HoverTool(renderers=[r2], tooltips=[(tags[2], '@y{0.00}')], mode='vline',attachment='above')
    p.add_tools(hover3)

    # add a hover tool for Normal
    hover4 = HoverTool(renderers=[r3], tooltips=[(tags[3], '@y{0.00}')], mode='vline',attachment='below')
    p.add_tools((hover4))

    hover5 = HoverTool(renderers=[r4], tooltips=[(tags[4], '@y{0.00}')], mode='vline',attachment='below')
    p.add_tools((hover5))


    p.legend.location = "top_left"
    p.legend.click_policy="mute"
    # p.y_range.start = -10
    # p.y_range.end = 10
    data = []
    

    log_csv = False
    prev_used = {}
    prev_used['exmem'] = None
    prev_used['normal'] = None
    prev_used['normal1'] = None
    
    p.xaxis.axis_label = 'Time (s)'
    p.yaxis.axis_label = YLABEL
    return p, ds0,ds1,ds2,ds3,ds4,dds0,dds1,dds2,dds3,dds4

tags =  ['CXL (MB)','Normal-0 (MB)','Normal-1 (MB)','CXL PMEM (MB)', 'GPU-MEM (MB)']
my_colors = ["#FFFF00","#FF1493","#00FF00","#00EFFF","#FF7F50"]
p,ds0,ds1,ds2,ds3,ds4,dds0,dds1,dds2,dds3,dds4 = get_p(tags, 'Memory Status [MemVerge]', 'Memory Used (MB)',my_colors)


    # Electric blue: "#00FFFF"
    # Bright green: "#00FF00"
    # Orange: "#FFA500"
    # Magenta: "#FF00FF"
    # Coral: "#FF7F50"
    # Yellow: "#FFFF00"
    # Deep pink: "#FF1493"
    # Lavender: "#E6E6FA"
    # Light gray: "#D3D3D3"

# my_colors = ["#00FFFF","#FF00FF","#FF7F50","#E6E6FA"]
tags =  ['GPU Util. (%)','GPU MEM Util. (%)','CPU Util. (%)','MEM Util. (%)', 'GPU Temp. (C)']
p2,ds02,ds12,ds22,ds32,ds42,dds02,dds12,dds22,dds32,dds42 = get_p(tags, 'Resources Utilization', 'Percent',my_colors)
grid = gridplot([[p],[p2]])
curdoc().add_root(grid)
# curdoc().add_root(p)
curdoc().theme = 'dark_minimal'
curdoc().title = 'System viewer'
duration_ms=1000
# Add a periodic callback to be run every 500 milliseconds
curdoc().add_periodic_callback(update, duration_ms)

