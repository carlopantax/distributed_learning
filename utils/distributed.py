import os
import logging
import torch.distributed as dist


def setup_logger(rank, log_dir='./logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_rank_{rank}.log')

    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - Rank %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - Rank %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        if is_main_process(rank):
            logger.addHandler(console_handler)

    return logger


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