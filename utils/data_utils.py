import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split

def split_train_val(dataset, val_ratio=0.2, seed=42):
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

def load_cifar100(root='./data', batch_size=128, distributed=False, rank=0, world_size=1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    full_trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    trainset, valset = split_train_val(full_trainset, val_ratio=0.2)

    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if distributed:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, valloader, testloader
