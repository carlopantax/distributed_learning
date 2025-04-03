import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if distributed:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader