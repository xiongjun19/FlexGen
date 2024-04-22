import matplotlib.pyplot as plt
log_file_path = "../OPT-66b-CXL-OUTPUT.log"
def extract_all_decode_throughputs(log_file_path):
    throughput_list = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'decode throughput' in line:
                throughput = float(line.split('decode throughput: ')[1].split(' token/s')[0])
                throughput_list.append(throughput)
    return throughput_list
cxl_decode_throughput= extract_all_decode_throughputs(log_file_path)
print(cxl_decode_throughput)
plt.plot(cxl_decode_throughput,'-o',label='CXL OFFLOAD -Decode throghput (tokens/s)')
plt.title('OPT-66B TEST SAMSUNG CXL MEMORY')
plt.xlabel('Test #')
plt.ylabel('Decode throughput (tokens/s)')
plt.ylim(0,6)
plt.savefig('cxl_perf.png')