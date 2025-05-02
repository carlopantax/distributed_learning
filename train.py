import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from utils.data_utils import load_cifar100
from models.lenet import LeNet5
import time
from config import learning_rate, batch_size, epochs
import os
from utils.plot_train import plot_data
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, setup_logger, get_rank, get_world_size
os.environ["OMP_NUM_THREADS"] = "4"  # Valore ottimale dipende dalla tua CPU
os.environ["MKL_NUM_THREADS"] = "4"  # Per Intel MKL

def train():
    """
    Environment Variables (set automatically by torchrun):
    - LOCAL_RANK: local GPU/process index on the current machine
    - RANK: unique global process ID
    - WORLD_SIZE: total number of processes (usually = number of GPUs)

    This script is designed for single-machine multi-process training with torchrun.

    Examples:
        !torchrun --nproc_per_node=4 train.py      # for 4-GPU or 4-CPU parallel training on one machine

    In order to excute centalized training, you can run the script as follows:
        !python train_centralized.py --optimizer sgdm --lr 0.05
        or !python train_centralized.py --optimizer adamw --lr 0.001

    To compare the results of centralized and distributed training, you can run the following command:
        !python compare_runs.py
    """
    # Reading torchrun variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    log_interval = 10

    setup_distributed(global_rank, world_size, backend=backend)

    logger = setup_logger(global_rank)

    if is_main_process(global_rank):
        logger.info(f"Starting training on single machine with {world_size} processes")
        logger.info(f"Device: {device} | Backend: {backend}")
        logger.info(f"Config: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")

    dist.barrier()

    trainloader, _ = load_cifar100(batch_size=batch_size, distributed=True,
                                   rank=global_rank, world_size=world_size)
    model = LeNet5(num_classes=100).to(device)

    if torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0

        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - start_time
                imgs_per_sec = (i + 1) * batch_size / elapsed
                logger.info(f"Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(trainloader)} | "
                            f"Loss: {avg_loss:.4f} | Images/sec: {imgs_per_sec:.2f}")
                running_loss = 0.0

        epoch_time = time.time() - epoch_start
        if is_main_process(global_rank):
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

        dist.barrier()

    if is_main_process(global_rank):
      torch.save(model.module.state_dict(), f'cifar100_model_ep{epochs}.pth')
      logger.info("Model checkpoint saved.")

      total_time = time.time() - start_time
      total_images = epochs * len(trainloader.dataset) * world_size  
      throughput = total_images / total_time


      logger.info(f"Training completed in {total_time:.2f} seconds.")
      logger.info(f"Average throughput: {throughput:.2f} Images/sec")


    cleanup_distributed()

    if is_main_process(global_rank):
        logger.info("Training complete.")


if __name__ == '__main__':
    train()
    plot_data() 
