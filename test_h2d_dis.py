# coding=utf8


import torch
import torch.distributed as dist
import numpy as np
import time


def test(device, rank, local_rank):
    shape=[4098, 256, 2048]
    torch_tensor = torch.randn(shape, pin_memory=True)
    # world_size = torch.distributed.get_world_size()
    # step = shape[-1] // world_size
    # split_ts = torch_tensor[:, :, step * local_rank:step *(local_rank+1)]
    
    iters=10
    start = time.time()
    for i in range(iters):
        device_mem = torch_tensor.to(device)
    end = time.time() 
    h2d_time = end - start
    print(f"H2D time: {h2d_time*1e3} ms")
    # Calculate bandwidth 
    bandwidth = (np.prod(shape)  * 4) / h2d_time / 1e6 * iters # MB/s
    
    print(f"the local_rank is {local_rank}; H2D time: {h2d_time*1e3} ms; Bandwidth: {bandwidth} MB/s")


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    device_num = torch.cuda.device_count()
    rank = dist.get_rank()
    local_rank = rank % device_num
    device = torch.device('cuda', local_rank)
    test(device, rank, local_rank)
