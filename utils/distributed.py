import os
import torch
import torch.distributed as dist

def setup_distributed(rank, world_size, backend='nccl', init_method='env://'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def get_world_size():
    return dist.get_world_size()

def get_rank():
    return dist.get_rank()