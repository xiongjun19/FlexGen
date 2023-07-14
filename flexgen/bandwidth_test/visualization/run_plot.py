# this script will generate a graph showing stats for all data in the csv files
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# TIME,CPU%,MEM%,GPU%,GPUMEM%,CXLMEM%,PCI_TX_MBps,PCI_RX_MBps,PRODUCT,PCI_INFO,GPUMEM_USED_MB, GPUMEM_TOTAL_MB

def moving_average(data, window_size):
    ma = data.rolling(window=window_size, min_periods=1).mean()
    return ma

# # Read the CSV files T4
# mem_df = pd.read_csv('mem.csv')
# cxl_df = pd.read_csv('cxl.csv')
# disk_df = pd.read_csv('disk.csv')
# memverge_df = pd.read_csv('output.csv')

mem_df = pd.read_csv('A10-12july/normal-b24.csv')
cxl_df = pd.read_csv('A10-12july/cxl-b24.csv')
disk_df = pd.read_csv('A10-12july/disk-b24-50perc.csv')
# memverge_df = pd.read_csv('A10-12july/memverge-b24.csv')

# Electric blue: "#00FFFF"
# Bright green: "#00FF00"
# Orange: "#FFA500"
# Magenta: "#FF00FF"
# Coral: "#FF7F50"
# Yellow: "#FFFF00"
# Deep pink: "#FF1493"
# Lavender: "#E6E6FA"
# Light gray: "#D3D3D3"



def plot_me(key):
    # Apply moving average filter
    window_size = 30
    colors = ['#FF4136', '#2ECC40', '#0074D9',"#FFA500"]
    # Extract the GPU utilization columns
    util_mem =  moving_average(mem_df[key],window_size)
    util_cxl = moving_average(cxl_df[key],window_size)
    util_disk = moving_average(disk_df[key],window_size)
    # gpu_util_memverge = moving_average(memverge_df[key],window_size)
    
    # Plotting
    plt.plot(util_mem[0:int(0.98*len(util_mem))], label='Normal0 Memory Offload',color=colors[0],alpha=0.9)
    plt.plot(util_cxl[0:int(0.98*len(util_cxl))], label='CXL Memory Offload',color=colors[1],alpha=0.9)
    plt.plot(util_disk[0:int(0.98*len(util_disk))], label='Disk NVMe Offload',color=colors[2],alpha=0.9) 
    # plt.plot(gpu_util_memverge[0:int(0.98*len(gpu_util_memverge))], label='Memverge',color=colors[3],alpha=0.9) 
    print('*'*80)  
    print(key)
    print('mem0 at 98% of length',util_mem[int(0.98*len(util_mem))])
    print('cxl at 98% of length',util_cxl[int(0.98*len(util_cxl))])
    print('disk 98% of length',util_disk[int(0.98*len(util_disk))])
    
    print('mem0 max',np.max(util_mem[:]))
    print('cxl max',np.max(util_cxl[:]))
    print('disk max',np.max(util_disk[:]))
    # print('memverge max',np.max(gpu_util_memverge[:]))
    
    print('mem0 mean',np.mean(util_mem[:]))
    print('cxl mean',np.mean(util_cxl[:]))
    print('disk mean',np.mean(util_disk[:]))
    # print('memverge mean',np.mean(gpu_util_memverge[:]))
    
    plt.xlabel('Time (s)', color="#00E5E4")  # Set x-axis label color to white
    plt.ylabel(f'{key}', color="#00E5E4")  # Set y-axis label color to white
    if key in ['GPU%']:
        plt.title(f'{key} - {GPU_INFO}', color="#00E5E4")  
    elif key in ['CXLMEM%']:
        plt.title(f'{key} - (TOTAL SIZE:{CXL_TOTAL_SIZE_GB}GB)', color="#00E5E4")  # Please put the actual size of CXL MB
    elif key in ['MEM%']:
        plt.title(f'{key} - (TOTAL SIZE:{TOTAL_SERVER_MEM_SIZE_GB}GB)', color="#00E5E4") 
    elif key in ['GPUMEM%']:
        plt.title(f'{key} - (TOTAL SIZE:{GPUMEM_TOTAL_GB}GB)', color="#00E5E4") 
    else:
        plt.title(f'{key}', color="#00E5E4")  # Set title color to white
    ax = plt.gca(facecolor='black')
    # Set the tick labels color to white
    ax.tick_params(axis='x', colors="#00E5E4")
    ax.tick_params(axis='y', colors="#00E5E4")
    
    ax.spines['top'].set_color("#00E5E4")
    ax.spines['bottom'].set_color("#00E5E4")
    ax.spines['left'].set_color("#00E5E4")
    ax.spines['right'].set_color("#00E5E4")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = plt.legend(loc='upper left')
    # Below are options to can use or leave empty for auto
    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    legend.get_frame().set_alpha(0.2)

if __name__=='__main__':
    GPU_INFO = mem_df['PRODUCT'][0]
    
    CXL_TOTAL_SIZE_GB = 128 # please update manually for SERVER CXL memory
    TOTAL_SERVER_MEM_SIZE_GB = 378 # please update manually for SERVER TOTAL memory= Normal0+Normal1+CXL
    if GPU_INFO=='NVIDIA A10':
        GPUMEM_TOTAL_GB = 24
    else: 
        assert 0, "Please check GPU MODEL and its enter Max GPUMEM_TOTAL_GB"
   
    fig = plt.figure(figsize=(12, 16),facecolor='black')
    plt.style.use('dark_background')
    fig.add_subplot(4,2,1)
    plot_me('CPU%')
    fig.add_subplot(4,2,2)
    plot_me('MEM%')
    fig.add_subplot(4,2,3)
    plot_me('GPU%')
    fig.add_subplot(4,2,4)
    plot_me('CXLMEM%')
    fig.add_subplot(4,2,5)
    plot_me('GPUMEM%')
    fig.add_subplot(4,2,6)
    plot_me('GPUMEM_USED_MB')
    fig.add_subplot(4,2,7)
    plot_me('PCI_TX_MBps')
    fig.add_subplot(4,2,8)
    plot_me('PCI_RX_MBps')

    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.55)

    plt.show()    
