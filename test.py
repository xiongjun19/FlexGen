# coding=utf8

from pynvml.smi import nvidia_smi 
nvsmi = nvidia_smi.getInstance()
res = nvsmi.DeviceQuery('memory.free, memory.total')
print(res)
print(len(res['gpu']))
