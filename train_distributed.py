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
import json
from collections import defaultdict
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process, setup_logger, get_rank, get_world_size
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

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

def train_distributed(args, tau=5, epochs_override=None):
    """
    Environment Variables (set automatically by torchrun):
    - LOCAL_RANK: local GPU/process index on the current machine
    - RANK: unique global process ID
    - WORLD_SIZE: total number of processes (usually = number of GPUs)

    This script is designed for single-machine multi-process training with torchrun.

    Examples:
        !torchrun --nproc_per_node=2 train_distributed.py --tau 4 --batch_size 128 --lr 0.05 --weight_decay 5e-4      # for 2-CPU parallel training on one machine

    In order to excute centalized training, you can run the script as follows:
        !python train_centralized.py --resume --optimizer sgdm --lr 0.01 --weight_decay 1e-4
        for SGDM: learning rate = [0.01, 0.05, 0.1], weight decay = [1e-4, 5e-4, 1e-3]
        for AdamW: learning rate = [1e-4, 3e-4, 1e-3], weight decay = [0.01, 0.05, 0.1]
)

    To compare the results of centralized and distributed training, you can run the following command:
        !python compare_runs.py
    """
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    log_interval = 10

    setup_distributed(global_rank, world_size, backend=backend)

    train_name = f"localSGD_lenet5_cifar100_w{world_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(
        log_dir='logs',
        train_name=train_name,
        rank=global_rank,
        is_main_process=is_main_process(global_rank),
        world_size=dist.get_world_size()
    )


    if is_main_process(global_rank):
        logger.log(f"Starting LocalSGD training on {world_size} workers")
        logger.log(f"Device: {device} | Backend: {backend}")
        logger.log(f"Config: batch_size={args.batch_size}, lr={args.lr}, epochs={epochs_override}")

    dist.barrier()

    trainloader, valloader, testloader, train_sampler  = load_cifar100(batch_size=args.batch_size, distributed=True, rank=global_rank, world_size=world_size)
    logger.local_train_size = len(trainloader.sampler)

    if is_main_process(global_rank):
        total_samples = len(trainloader.dataset)  # 40.000 in CIFAR100 80%
        samples_per_worker = len(train_sampler) if train_sampler else total_samples
        logger.log(f"[INFO] Train set size (total): {total_samples}")
        logger.log(f"[INFO] Each worker sees approximately {samples_per_worker} samples per epoch.")


    model = LeNet5(num_classes=100).to(device)

    model = DistributedDataParallel(model, device_ids=[local_rank]) if torch.cuda.is_available() else DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    num_epochs = epochs_override if epochs_override is not None else epochs
    total_steps = num_epochs
    milestones = [int(0.5 * total_steps), int(0.75 * total_steps)]

    warmup_epochs = 20
    def warmup_lr(epoch):
        return float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    from torch.optim.lr_scheduler import LambdaLR
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lr)

    scheduler_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


    start_epoch = 0
    best_val_acc = 0.0

    checkpoint_path = f'checkpoints/local_sgd_model_{world_size}.pth'
    if args.resume and os.path.exists(checkpoint_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.module.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        if is_main_process(global_rank):
            logger.log(f"Resumed training from checkpoint at epoch {start_epoch}")

    start_time = time.time()
    total_train_size = len(trainloader.dataset) * world_size

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        correct = total = 0
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
            scheduler_decay.step()  # Decay (MultiStepLR)

            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % tau == 0:
                for param in model.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    param.data /= world_size

            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                train_acc = 100.0 * correct / total if total > 0 else 0
                logger.log_metrics(
                    epoch=epoch,
                    batch_idx=i,
                    loss=avg_loss,
                    batch_size=args.batch_size,
                    train_size=total_train_size,
                    extras={'lr': optimizer.param_groups[0]['lr'], 'train_acc': train_acc}
                )
                running_loss = 0.0
        scheduler_warmup.step()  # Warmup (LambdaLR)
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / batch_count
        train_acc = 100.0 * correct / total if total > 0 else 0
        val_acc = compute_accuracy(model.module, valloader, device)

        logger.log_epoch(
            epoch=epoch,
            epoch_loss=avg_epoch_loss,
            epoch_time=epoch_time,
            world_size=world_size,
            extras={'train_acc': train_acc, 'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']}
        )

        if is_main_process(global_rank):
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_acc': max(best_val_acc, val_acc)
            }, checkpoint_path)
            best_val_acc = max(best_val_acc, val_acc)

        dist.barrier()

    total_time = time.time() - start_time
    total_images = num_epochs * total_train_size
    logger.log_training_complete(total_time, total_images, world_size)

    if is_main_process(global_rank):
        # Merge log files da tutti i rank
        merged_log_file = os.path.join('logs', f"{train_name}_all.log")
        log_entries = []

        for r in range(world_size):
            rank_log_path = os.path.join('logs', f"{train_name}_rank{r}.log")
            if os.path.exists(rank_log_path):
                with open(rank_log_path, 'r') as infile:
                    for line in infile:
                        try:
                            timestamp_str = line.split(' - ')[0]
                            timestamp = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                            log_entries.append((timestamp, line))
                        except Exception:
                            log_entries.append(((9999,), line))  # fallback in caso di parsing errato

        log_entries.sort(key=lambda x: x[0])
        with open(merged_log_file, 'w') as outfile:
            for _, line in log_entries:
                outfile.write(line)

        logger.log(f"All logs merged to {merged_log_file}")

        # Generazione plot per ciascun rank
        for r in range(world_size):
            metrics_path = os.path.join('logs', f"{train_name}_metrics_rank{r}.json")
            if os.path.exists(metrics_path):
                plotter = TrainingPlotter(
                    log_dir='logs',
                    train_name=f"{train_name}_rank{r}",
                    is_main_process=True,
                    metrics_files=[metrics_path]
                )
                logger.log(f"Plotting metrics for rank {r}...")
                plotter.plot_all_per_rank()

        logger.log("All plots generated.")

        # Merge dei metrics JSON
        ranked_metrics = {}
        for r in range(world_size):
            metrics_path = os.path.join('logs', f"{train_name}_metrics_rank{r}.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    ranked_metrics[f"rank{r}"] = json.load(f)

        merged_metrics_path = os.path.join('logs', f"{train_name}_metrics_combined.json")
        with open(merged_metrics_path, 'w') as f:
            json.dump(ranked_metrics, f, indent=4)

        logger.log(f"All metrics merged with rank info to {merged_metrics_path}")



    cleanup_distributed()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=int, default=5, help='Local SGD steps before sync')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.001, help = 'Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    args = parser.parse_args()

    dataset_size = 50000
    epochs_central = epochs
    total_batches_central = (dataset_size / args.batch_size) * epochs_central
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    epochs_local = int(total_batches_central / (dataset_size / (args.batch_size * world_size)))

    train_distributed(args, tau=args.tau, epochs_override=epochs_local)