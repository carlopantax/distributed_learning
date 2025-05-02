import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.lenet import LeNet5
from utils.data_utils import load_cifar100
from config import batch_size, epochs


def setup_logger(log_path='logs_centralized/training_centralized.log'):
    logger = logging.getLogger('centralized')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def train_centralized(optimizer_type='sgdm', learning_rate=0.001, resume=True, checkpoint_path='checkpoint.pth'):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training centralized on device: {device}")
    logger.info(f"Optimizer: {optimizer_type.upper()}, Epochs: {epochs}, LR: {learning_rate}")

    trainloader, valloader, _ = load_cifar100(batch_size=batch_size, distributed=False)

    model = LeNet5(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if optimizer_type == 'sgdm':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    else:
        raise ValueError("Invalid optimizer type. Choose 'sgdm' or 'adamw'.")

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_acc = 0.0

    # Load checkpoint if exists
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0
        epoch_start = time.time()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc = 100.0 * correct / total
        val_acc = compute_accuracy(model, valloader, device)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(valloader)

        duration = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}% | Duration: {duration:.2f}s")

        # Save checkpoint
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_acc': max(best_val_acc, val_acc)
        }
        torch.save(state, checkpoint_path)

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds.")

    model_path = f'model_centralized_{optimizer_type}.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='sgdm', choices=['sgdm', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    args = parser.parse_args()

    train_centralized(
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )
