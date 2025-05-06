import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch.distributed as dist

from utils.TrainingLogger import TrainingLogger
from utils.TrainingPlotter import TrainingPlotter
from utils.data_utils import load_cifar100
from models.lenet import LeNet5
import time
from config import learning_rate, batch_size, epochs
import os
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
        !python train_centralized.py --resume --optimizer sgdm --lr 0.01 --weight_decay 1e-4
        for SGDM: learning rate = [0.01, 0.05, 0.1], weight decay = [1e-4, 5e-4, 1e-3]
        for AdamW: learning rate = [1e-4, 3e-4, 1e-3], weight decay = [0.01, 0.05, 0.1]
)

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

    # logger = setup_logger(global_rank)
    train_name = f"lenet5_cifar100_w{world_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(
        log_dir='logs',
        train_name=train_name,
        rank=global_rank,
        is_main_process=is_main_process(global_rank)
    )

    if is_main_process(global_rank):
        logger.log(f"Starting training on single machine with {world_size} processes")
        logger.log(f"Device: {device} | Backend: {backend}")
        logger.log(f"Config: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")

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
    total_train_size = len(trainloader.dataset) * world_size

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0

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
            epoch_loss += loss.item()
            batch_count += 1


            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                logger.log_metrics(
                    epoch=epoch,
                    batch_idx=i,
                    loss=avg_loss,
                    batch_size=batch_size,
                    train_size=total_train_size,
                    extras={'lr': optimizer.param_groups[0]['lr']}
                )
                running_loss = 0.0


        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / batch_count

        # Log epoch stats with our new logger
        logger.log_epoch(
            epoch=epoch,
            epoch_loss=avg_epoch_loss,
            epoch_time=epoch_time,
            world_size=world_size
        )

        dist.barrier()

    if is_main_process(global_rank):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/cifar100_model_ep{epochs}.pth'
        torch.save(model.module.state_dict(), checkpoint_path)
        logger.log(f"Model checkpoint saved to {checkpoint_path}")

        total_time = time.time() - start_time
        total_images = epochs * total_train_size

        logger.log_training_complete(total_time, total_images, world_size)

    cleanup_distributed()

    if is_main_process(global_rank):
        logger.log("Generating training plots...")
        plotter = TrainingPlotter(
            log_dir='logs',
            train_name=train_name,
            is_main_process=True
        )
        plotter.plot_all()
        logger.log("Training complete. Plots generated.")


if __name__ == '__main__':
    train()