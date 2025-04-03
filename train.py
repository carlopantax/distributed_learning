import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size
from utils.data_utils import load_cifar100
from models.cnn import CNN
from config import learning_rate, batch_size, epochs
import os

def train():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    global_rank = int(os.environ['RANK'])

    device = torch.device("cpu") # Set device to CPU

    setup_distributed(global_rank, world_size, backend='gloo', init_method='env://') # Use 'gloo' backend for CPU

    trainloader, _ = load_cifar100(batch_size=batch_size, distributed=True, rank=global_rank, world_size=world_size)
    model = CNN(num_classes=100).to(device)
    model = DistributedDataParallel(model) # No device_ids needed for CPU
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
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