import torch.distributed as dist
import logging


def setup_distributed(rank, world_size, backend='nccl', init_method='env://'):
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=init_method)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def get_world_size():
    return dist.get_world_size()


def get_rank():
    return dist.get_rank()


def setup_logger(rank, log_dir='./logs'):
    import os
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_rank_{rank}.log')

    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - [Rank %(name)s] - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if is_main_process(rank):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
