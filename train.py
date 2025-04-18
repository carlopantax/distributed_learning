import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size, setup_logger
from utils.data_utils import load_cifar100
from models.lenet import LeNet5
import time
from config import learning_rate, batch_size, epochs
import os
from utils.plot_train import plot_data
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Valore ottimale dipende dalla tua CPU
os.environ["MKL_NUM_THREADS"] = "4"  # Per Intel MKL

def train():
    """
    Environment Variables:
    LOCAL_RANK, WORLD_SIZE, and RANK are typically set by a launcher like torchrun.
    WORLD_SIZE: total number of processes (often same as number of devices).
    RANK: unique ID for each process.
    LOCAL_RANK: local device ID on the node (e.g. GPU 0 or 1).

    setup_distributed(global_rank, world_size, backend='gloo', init_method='env://')  ---> For CPU
    setup_distributed(global_rank, world_size, backend='nccl', init_method='env://')  ---> For GPU

    PC A:
    MASTER_ADDR=192.168.1.10 MASTER_PORT=12355 RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 python train.py

    PC B:
    MASTER_ADDR=192.168.1.10 MASTER_PORT=12355 RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 python train.py

    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12355 train.py

    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.10 --master_port=12355 train.py
    """
    # Get distributed training parameters from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))

    # Set up training parameters
    device = torch.device("cpu")

    # Log every X batches
    log_interval = 10

    backend = 'gloo'  # 'nccl' for GPU
    setup_distributed(global_rank, world_size, backend=backend, init_method='env://')

    # Set up logging
    logger = setup_logger(global_rank)

    # Log training configuration
    if is_main_process(global_rank):
        logger.info(f"Starting distributed training with {world_size} processes")
        logger.info(f"Training configuration: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")
        logger.info(f"Using {backend} backend for distributed communication")

    # Synchronize processes before starting
    dist.barrier()

    # Load data
    try:
        trainloader, testloader = load_cifar100(batch_size=batch_size, distributed=True,
                                                rank=global_rank, world_size=world_size)
        logger.info(f"Data loaded successfully on rank {global_rank}")
    except Exception as e:
        logger.error(f"Error loading data on rank {global_rank}: {str(e)}")
        cleanup_distributed()
        return

    try:
        model = LeNet5(num_classes=100).to(device)
        model = DistributedDataParallel(model)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logger.info(f"Model initialized on rank {global_rank}")
    except Exception as e:
        logger.error(f"Error initializing model on rank {global_rank}: {str(e)}")
        cleanup_distributed()
        return

    # Training loop
    start_time = time.time()
    global_step = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        # Set epoch for distributed sampler
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)

        for i, (inputs, labels) in enumerate(trainloader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                global_step += 1

                # Log every log_interval mini-batches
                if i % log_interval == log_interval - 1:
                    avg_loss = running_loss / log_interval
                    elapsed = time.time() - start_time
                    imgs_per_sec = (i + 1) * batch_size / elapsed

                    logger.info(
                        f"Epoch: {epoch + 1}/{epochs} | "
                        f"Batch: {i + 1}/{len(trainloader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Images/sec: {imgs_per_sec:.2f}"
                    )
                    running_loss = 0.0

            except Exception as e:
                logger.error(f"Error during training step on rank {global_rank}, batch {i}: {str(e)}")
                continue

        # Calculate and log epoch statistics
        epoch_time = time.time() - epoch_start_time

        # Only main process logs overall epoch summary
        if is_main_process(global_rank):
            logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds | "
                f"Total time elapsed: {time.time() - start_time:.2f} seconds"
            )

        # Synchronize all processes after each epoch
        dist.barrier()

    # Save model checkpoint (only on main process)
    if is_main_process(global_rank):
        try:
            checkpoint_path = f'cifar100_distributed_model_epoch_{epochs}.pth'
            torch.save(model.module.state_dict(), checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    # Calculate and log total training time
    total_time = time.time() - start_time
    logger.info(f"Rank {global_rank} completed training in {total_time:.2f} seconds")

    # Wait for all processes to finish
    dist.barrier()
    cleanup_distributed()

    if is_main_process(global_rank):
        logger.info("Distributed training completed successfully")


if __name__ == '__main__':
    train()
    #plot_data()