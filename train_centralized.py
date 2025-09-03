import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from lars import LARS
from lamb import LAMB
from torch.optim.lr_scheduler import LambdaLR
import math
from models.lenet import LeNet5
from utils.TrainingLogger import TrainingLogger
from utils.TrainingPlotter import TrainingPlotter
from utils.data_utils import load_cifar100

def setup_logger(log_path='logs_centralized/training_centralized.log'):
    """
    Create a process-wide logger that writes to both a file and stdout.

    Args:
        log_path: Destination path for the log file. Parent dirs are created.

    Returns:
        A configured `logging.Logger` named 'centralized'.
    """
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
    """
    Compute top-1 accuracy of a model over a dataloader.

    Args:
        model: Torch module in eval or train mode.
        dataloader: Iterable of (inputs, labels).
        device: 'cpu' or 'cuda' device where tensors will be moved.

    Returns:
        Accuracy percentage.
    """
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

def scale_lr(base_lr, base_batch_size, actual_batch_size, mode='sqrt'):
    """
    Scale a base learning rate when changing the mini-batch size.

    Args:
        base_lr: Reference LR tuned at `base_batch_size`.
        base_batch_size: Batch size used to tune `base_lr`.
        actual_batch_size: Current batch size.
        mode: 'linear' (kx) or 'sqrt' (√kx) scaling; √k is capped to avoid extremes.

    Returns:
        Scaled learning rate.
    """
    scale = actual_batch_size / base_batch_size
    if mode == 'linear':
        return base_lr * scale
    elif mode == 'sqrt':
        max_scale=50
        return base_lr * min(scale**0.5, max_scale)
    else:
        raise ValueError("Unsupported scaling mode.")

def warmup_cosine_schedule(epoch, warmup_epochs=40, total_epochs=150):
    """
    Piecewise LR multiplier: linear warm-up then cosine decay to 0.

    Args:
        epoch: Current epoch index (0-based).
        warmup_epochs: Number of warm-up epochs.
        total_epochs: Total planned epochs (defines cosine horizon).

    Returns:
        A scalar multiplier in [0, 1] applied by LambdaLR.
    """
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * (1. + math.cos(math.pi * progress))


def train_centralized(optimizer_type='sgdm', learning_rate=0.001, weight_decay=1e-4, resume=True, checkpoint_path='./checkpoint.pth', batch_size=128, epochs=150):
    """
    Train LeNet-5 on CIFAR-100 in a single process with rich logging.

    What it does:
      - Builds CIFAR-100 loaders with standard normalization/augmentation.
      - Instantiates LeNet-5 and a chosen optimizer: SGDM, AdamW, LARS, or LAMB.
      - Optionally rescales LR for large batches (√k for LARS/LAMB).
      - Uses warmup+cosine scheduling via `LambdaLR`.
      - Resumes training state from `checkpoint_path` if requested and present.
      - Logs per-batch and per-epoch metrics to `TrainingLogger` JSON files.
      - Saves rolling checkpoints each epoch and the final model at the end.
      - Generates plots via `TrainingPlotter`.

    Args:
        optimizer_type: 'sgdm' | 'adamw' | 'lars' | 'lamb'.
        learning_rate: Base LR (may be scaled for LARS/LAMB).
        weight_decay: L2 weight decay (coupled for SGDM/AdamW; see optimizers).
        resume: If True, try to load model/optimizer/scheduler state.
        checkpoint_path: Path to read/write checkpoint dict.
        batch_size: Training batch size.
        epochs: Total number of training epochs.
    """
    base_name = f"centralized_{optimizer_type}_lr{learning_rate}_wd{weight_decay}_batch_size{batch_size}"
    if resume and os.path.exists(checkpoint_path):
        DIR = f"{base_name}"
        existing = [f for f in os.listdir(DIR) if f.endswith('_metrics.json')]
        if existing:
            train_name = existing[0].replace('_metrics.json', '')
        else:
            train_name = f"{base_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        train_name = f"{base_name}_{timestamp}"
        DIR = f"{base_name}"
    logger = TrainingLogger(
        log_dir=DIR,
        train_name=train_name
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Training centralized on device: {device}")
    logger.log(f"Optimizer: {optimizer_type.upper()}, Epochs: {epochs}, LR: {learning_rate}")

    trainloader, valloader, _, _ = load_cifar100(batch_size=batch_size)
    logger.log(f"[DEBUG] Train dataset size: {len(trainloader.dataset)} | Expected batches: {len(trainloader)}")
    model = LeNet5(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    base_batch_size = 128
    scaling_mode = 'sqrt' if optimizer_type in ['lars', 'lamb'] else None

    if scaling_mode:
      learning_rate = scale_lr(learning_rate, base_batch_size, batch_size, mode=scaling_mode)
      logger.log(f"Scaled LR using {scaling_mode} mode to: {learning_rate}")
    if optimizer_type == 'sgdm':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'lars':
      optimizer = LARS(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'lamb':
      optimizer = LAMB(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer type. Choose 'sgdm', 'adamw', 'lars', 'lamb'.")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_cosine_schedule(epoch, warmup_epochs=20, total_epochs=epochs))

    start_epoch = 0
    best_val_acc = 0.0

    logger.log(f"Checking if there is any checkpoint. Resuming {resume}, Checkpoint path: {checkpoint_path}")
    if resume and os.path.exists(checkpoint_path):
        logger.log(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)

    start_time = time.time()
    log_interval = 10
    train_size = len(trainloader.dataset)
    total_batches = len(trainloader)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        correct = total = 0
        epoch_start = time.time()
        batch_count = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                current_lr = optimizer.param_groups[0]['lr']

                logger.log_metrics(
                    epoch=epoch,
                    batch_idx=i,
                    loss=avg_loss,
                    batch_size=batch_size,
                    train_size=train_size,
                    extras={
                        'lr': current_lr,
                        'train_acc': 100.0 * correct / total if total > 0 else 0
                    }
                )
                running_loss = 0.0

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

        epoch_time = time.time() - epoch_start
        logger.log_epoch(
            epoch=epoch,
            epoch_loss=epoch_loss / batch_count if batch_count > 0 else 0,
            epoch_time=epoch_time,
            extras={
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            }
        )
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_acc': max(best_val_acc, val_acc)
        }
        torch.save(state, checkpoint_path)
        best_val_acc = max(best_val_acc, val_acc)

    total_time = time.time() - start_time
    total_images = epochs * train_size
    logger.log_training_complete(total_time, total_images)

    model_path = f'model_centralized_{optimizer_type}.pth'
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved to {model_path}")

    logger.log("Generating training plots...")

    rank0_file = os.path.join(DIR, f"{train_name}_metrics_rank0.json")
    default_file = os.path.join(DIR, f"{train_name}_metrics.json")
    if os.path.exists(rank0_file):
        import shutil
        shutil.copy(rank0_file, default_file)

    plotter = TrainingPlotter(
        log_dir=DIR,
        train_name=train_name
    )

    plotter.plot_all()
    logger.log("Training complete. Plots generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='sgdm', choices=['sgdm', 'adamw', 'lars', 'lamb'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--batch_size', type = int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()

    train_centralized(
      optimizer_type=args.optimizer,
      learning_rate=args.lr,
      resume=args.resume,
      weight_decay=args.weight_decay,
      checkpoint_path=args.checkpoint,
      batch_size=args.batch_size,
      epochs=args.epochs
)

