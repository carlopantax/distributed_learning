import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size
from utils.data_utils import load_cifar100  # Import the correct loading function
from models.cnn import CNN
from config import learning_rate, batch_size, epochs

def train(rank, world_size):
    setup_distributed(rank, world_size)

    trainloader, _ = load_cifar100(batch_size=batch_size, distributed=True, rank=rank, world_size=world_size) # Use load_cifar100
    model = SimpleCNN(num_classes=100).to(rank)  # Instantiate with 100 output classes
    model = DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and is_main_process(rank):
                print(f"Rank: {rank}, Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item():.4f}")

    if is_main_process(rank):
        print("Finished Training on CIFAR-100")
        # Save the model with a descriptive name
        torch.save(model.module.state_dict(), 'cifar100_distributed_model.pth')

    cleanup_distributed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Training on CIFAR-100")
    parser.add_argument('--world_size', type=int, default=2, help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=-1, help='Rank of the current process')
    args = parser.parse_args()

    if args.rank == -1:
        torch.multiprocessing.spawn(train, args=(args.world_size,), nprocs=args.world_size, join=True)
    else:
        train(args.rank, args.world_size)