import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.lenet import LeNet5
from utils.data_utils import load_cifar100
from config import learning_rate, batch_size, epochs


def setup_logger(log_path='logs_centralized/training_centralized.log'):
    logger = logging.getLogger('centralized')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def train_centralized():
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training centralized on device: {device}")

    # Load data (no DistributedSampler)
    trainloader, _ = load_cifar100(batch_size=batch_size, distributed=False)

    model = LeNet5(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        duration = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Duration: {duration:.2f}s")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds.")

    total_images = epochs * len(trainloader.dataset)
    throughput = total_images / total_time
    logger.info(f"Average throughput: {throughput:.2f} Images/sec")

    model_path = 'model_centralized.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == '__main__':
    train_centralized()
