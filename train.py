import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size
from utils.data_utils import load_cifar100
from models.cnn import CNN
from config import learning_rate, batch_size, epochs
import os

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
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    global_rank = int(os.environ['RANK'])

    device = torch.device("cpu")

    setup_distributed(global_rank, world_size, backend='gloo', init_method='env://')

    trainloader, _ = load_cifar100(batch_size=batch_size, distributed=True, rank=global_rank, world_size=world_size)
    model = CNN(num_classes=100).to(device)
    model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and is_main_process(global_rank):
                print(f"Rank: {global_rank}, Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item():.4f}")

    if is_main_process(global_rank):
        print("Finished Training on CIFAR-100 (CPU)")
        torch.save(model.module.state_dict(), 'cifar100_distributed_cpu_model.pth')

    cleanup_distributed()

if __name__ == '__main__':
    train()