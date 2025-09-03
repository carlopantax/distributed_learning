import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def split_train_val(dataset, val_ratio=0.2, seed=42):
    """
    Split a dataset into training and validation subsets.

    Args:
        dataset: The dataset to be split.
        val_ratio: Fraction of the dataset to use for validation.
        seed: Random seed for reproducibility.

    Returns:
            - Training subset
            - Validation subset
    """
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

def load_cifar100_for_clients(num_clients=4, batch_size=128, root='./data', seed=42):
    """
    Load the CIFAR-100 dataset and partition it into subsets for multiple clients,
    simulating a distributed learning setting.

    Args:
        num_clients: Number of clients to split the dataset across.
        batch_size: Batch size for each client DataLoader.
        root: Directory where the CIFAR-100 data will be stored/downloaded.
        seed: Random seed for reproducibility.

    Returns:
            - client_loaders: Training DataLoaders, one for each client.
            - client_val_loaders: Validation DataLoaders, one for each client.
            - testloader: Global test DataLoader for evaluation.
    """
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

    trainset, valset = split_train_val(full_trainset, val_ratio=0.2, seed=seed)

    # Split training set
    train_size_per_client = len(trainset) // num_clients
    train_sizes = [train_size_per_client] * num_clients
    for i in range(len(trainset) % num_clients):
        train_sizes[i] += 1

    generator = torch.Generator().manual_seed(seed)
    client_train_subsets = random_split(trainset, train_sizes, generator=generator)

    client_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for subset in client_train_subsets
    ]

    # Split validation set
    val_size_per_client = len(valset) // num_clients
    val_sizes = [val_size_per_client] * num_clients
    for i in range(len(valset) % num_clients):
        val_sizes[i] += 1

    client_val_subsets = random_split(valset, val_sizes, generator=generator)

    client_val_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        for subset in client_val_subsets
    ]

    # Global test set 
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return client_loaders, client_val_loaders, testloader

def load_cifar100(root='./data', batch_size=128):
    """
    Load CIFAR-100 dataset with train, validation, and test splits.

    Args:
        root: Directory where the CIFAR-100 data will be stored/downloaded.
        batch_size: Batch size for DataLoaders.

    Returns:
            - trainloader : DataLoader for the training set.
            - valloader : DataLoader for the validation set.
            - testloader : DataLoader for the test set.
            - train_sampler : Distributed sampler if used, otherwise None.
    """
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

    train_sampler = None
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, valloader, testloader, train_sampler
