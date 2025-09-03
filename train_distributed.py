import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from utils.TrainingLogger import TrainingLogger
from utils.TrainingPlotter import TrainingPlotter
from utils.data_utils import load_cifar100_for_clients as load_cifar100
from models.lenet import LeNet5
import time
from utils.outer_optimizer_bmuf import initialize_global_momentum, global_momentum_update
from utils.slowmo import initialize_slowmo_state, slowmo_update
import os
import json
import math
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def compute_accuracy(model, dataloader, device):
    """
    Compute top-1 accuracy on a given dataloader.

    Args:
        model: Torch model to evaluate.
        dataloader: Iterable of (inputs, labels).
        device: Device where tensors/model live.

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

def average_models(models):
    """
    Distributed Avg-style parameter averaging across client models.

    Args:
        models: List of torch.nn.Module with identical structure.

    Returns:
        A state_dict containing the elementwise mean of parameters.
    """
    avg_state_dict = copy.deepcopy(models[0].state_dict())
    for key in avg_state_dict:
        for i in range(1, len(models)):
            avg_state_dict[key] += models[i].state_dict()[key]
        avg_state_dict[key] = avg_state_dict[key] / len(models)
    return avg_state_dict

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr):
    """
    Cosine LR schedule wrapped by LambdaLR with a linear warm-up phase.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warm-up epochs.
        total_epochs: Total epochs for cosine horizon.
        base_lr: Unused here (kept for signature symmetry).

    Returns:
        torch.optim.lr_scheduler.LambdaLR instance.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = float(current_epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_distributed(args, epochs_override=None):
    """
    Simulate LocalSGD with sequential clients and periodic model averaging.

    Workflow:
      - Split CIFAR-100 into `world_size` client loaders.
      - For each round: each client trains `tau` local epochs starting from the
        current global weights; per-batch/epoch metrics are logged.
      - After all clients finish a round, average their weights.
      - Optionally apply a global outer optimizer:
          * BMUF-style momentum via `global_momentum_update`
          * SlowMo-style update via `slowmo_update`
      - Save global checkpoints and, at the end, plot metrics per client.

    Args:
        args: Namespace containing training hyperparameters
              (tau, batch_size, lr, weight_decay, world_size, epochs, etc.).
        epochs_override: args.epochs.

    """
    num_clients = args.world_size
    tau = args.tau

    train_name = f"localSGD_lenet5_seq_cifar100_w{num_clients}_{time.strftime('%Y%m%d_%H%M%S')}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    general_logger = TrainingLogger(
    log_dir='logs',
    train_name=train_name,
    client_id=100,
    num_clients=num_clients
    )

    general_logger.log(f"Simulating LocalSGD training with {num_clients} clients (sequential execution)")
    general_logger.log(f"Device: {device}")
    general_logger.log(f"Config: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}, tau={tau}")

    client_loaders, client_val_loaders, testloader = load_cifar100(
        batch_size=args.batch_size,
        num_clients=num_clients
    )

    total_train_size = sum(len(loader.dataset) for loader in client_loaders)
    general_logger.log(f"[INFO] Total train set size: {total_train_size}")

    num_epochs = epochs_override if epochs_override is not None else args.epochs
    num_rounds = num_epochs // tau
    remainder = num_epochs % tau

    best_val_acc = 0.0
    global_model = LeNet5(num_classes=100).to(device)
    os.makedirs('checkpoints', exist_ok=True)
    momentum_buffer = None
    slowmo_buffer = None
    if args.use_global_momentum:
        momentum_buffer = initialize_global_momentum(global_model)
    elif args.use_slowmo:
        slowmo_buffer = initialize_slowmo_state(global_model)



    for round_idx in range(num_rounds):
        epoch_loss_total = 0.0
        total_seen_all_clients = 0
        round_start_time = time.time()
        general_logger.log(f"--- Round {round_idx + 1}/{num_rounds} (tau = {tau}) ---")
        client_models = []

        for client_idx in range(num_clients):
            model = LeNet5(num_classes=100).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=num_epochs, base_lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            logger = TrainingLogger(
                log_dir='logs',
                train_name=train_name,
                client_id=client_idx,
                num_clients=num_clients
            )

            for local_epoch in range(tau):
                model.train()
                batch_start_time = time.time()
                running_correct = 0
                total_seen = 0
                running_loss = 0.0
                total_batches = len(client_loaders[client_idx])
                log_freq = 10  

                for batch_idx, (inputs, labels) in enumerate(client_loaders[client_idx], start=1):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_correct += (preds == labels).sum().item()
                    total_seen += inputs.size(0)

                    if batch_idx % log_freq == 0 or batch_idx == total_batches:
                        current_time = time.time()
                        elapsed_time = current_time - batch_start_time
                        images_per_sec = total_seen / elapsed_time if elapsed_time > 0 else 0.0
                        progress = 100.0 * batch_idx / total_batches
                        avg_loss = running_loss / total_seen
                        train_acc = 100.0 * running_correct / total_seen

                        logger.log(
                            f"Client {client_idx} | Epoch {local_epoch+1} | Batch {batch_idx}/{total_batches} | "
                            f"Loss: {avg_loss:.4f} | Images/sec: {images_per_sec:.2f} | "
                            f"Progress: {progress:.1f}% | Time: {elapsed_time:.2f}s | "
                            f"lr: {args.lr} | train_acc: {train_acc:.4f}"
                        )
                        logger.log_metrics(
                            epoch=local_epoch,
                            batch_idx=batch_idx,
                            loss=avg_loss,
                            batch_size=inputs.size(0),
                            train_size=len(client_loaders[client_idx].dataset),
                            extras={'train_acc': train_acc, 'lr': scheduler.get_last_lr()[0]}
                        )

                general_logger.log(f"[Client {client_idx}] Completed local epoch {local_epoch+1}/{tau} | "
                            f"Final Loss: {running_loss / total_seen:.4f} | Final Train Acc: {100.0 * running_correct / total_seen:.2f}%")
                
                val_acc = compute_accuracy(model, client_val_loaders[client_idx], device)
                epoch_loss_total += running_loss
                total_seen_all_clients += total_seen
                client_models.append(model)

                current_epoch = (round_idx + 1) * tau
                train_acc = compute_accuracy(model, torch.utils.data.DataLoader(client_loaders[client_idx].dataset, batch_size=128), device)

                epoch_time = time.time() - round_start_time
                avg_epoch_loss = epoch_loss_total / total_seen_all_clients
                logger.log_epoch(
                    epoch=current_epoch,
                    epoch_loss=avg_epoch_loss,
                    epoch_time=epoch_time,
                    world_size=num_clients,
                    extras={'train_acc': train_acc, 'val_acc': val_acc, 'lr': scheduler.get_last_lr()[0]}
                )
                scheduler.step()


        averaged_weights = average_models(client_models)

        if args.use_global_momentum:
            updated_weights, momentum_buffer = global_momentum_update(global_model, averaged_weights, momentum_buffer)
            global_model.load_state_dict(updated_weights)
        elif args.use_slowmo:
            updated_weights, slowmo_buffer = slowmo_update(global_model, averaged_weights, slowmo_buffer)
            global_model.load_state_dict(updated_weights)
        else:
            global_model.load_state_dict(averaged_weights)


        
        torch.save(global_model.state_dict(), f'checkpoints/global_model_epoch{current_epoch}.pth')

    if remainder > 0:
        epoch_loss_total = 0.0
        total_seen_all_clients = 0
        round_start_time = time.time()
        general_logger.log(f"--- Final round (tau = {remainder}) ---")
        client_models = []

        for client_idx in range(num_clients):
            logger = TrainingLogger(
                log_dir='logs',
                train_name=train_name,
                client_id=client_idx,
                num_clients=num_clients
            )

            model = LeNet5(num_classes=100).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_epochs=5, total_epochs=num_epochs, base_lr=args.lr
            )
            criterion = nn.CrossEntropyLoss()

            total_seen = 0
            running_loss = 0.0

            for local_epoch in range(remainder):
                model.train()
                batch_start_time = time.time()
                running_correct = 0
                total_seen = 0
                running_loss = 0.0
                total_batches = len(client_loaders[client_idx])
                log_freq = 10

                for batch_idx, (inputs, labels) in enumerate(client_loaders[client_idx], start=1):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_correct += (preds == labels).sum().item()
                    total_seen += inputs.size(0)

                    if batch_idx % log_freq == 0 or batch_idx == total_batches:
                        current_time = time.time()
                        elapsed_time = current_time - batch_start_time
                        images_per_sec = total_seen / elapsed_time if elapsed_time > 0 else 0.0
                        progress = 100.0 * batch_idx / total_batches
                        avg_loss = running_loss / total_seen
                        train_acc = 100.0 * running_correct / total_seen

                        logger.log(
                            f"Client {client_idx} | Epoch {local_epoch+1} | Batch {batch_idx}/{total_batches} | "
                            f"Loss: {avg_loss:.4f} | Images/sec: {images_per_sec:.2f} | "
                            f"Progress: {progress:.1f}% | Time: {elapsed_time:.2f}s | "
                            f"lr: {args.lr} | train_acc: {train_acc:.4f}"
                        )

                        logger.log_metrics(
                            epoch=local_epoch,
                            batch_idx=batch_idx,
                            loss=avg_loss,
                            batch_size=inputs.size(0),
                            train_size=len(client_loaders[client_idx].dataset),
                            extras={'train_acc': train_acc, 'lr': scheduler.get_last_lr()[0]}
                        )

                logger.log(
                    f"[Client {client_idx}] Completed local epoch {local_epoch+1}/{remainder} | "
                    f"Final Loss: {running_loss / total_seen:.4f} | Final Train Acc: {100.0 * running_correct / total_seen:.2f}%"
                )

                val_acc = compute_accuracy(model, client_val_loaders[client_idx], device)
                train_acc = compute_accuracy(model, torch.utils.data.DataLoader(client_loaders[client_idx].dataset, batch_size=128), device)

                client_models.append(model)
                epoch_loss_total += running_loss
                total_seen_all_clients += total_seen

                epoch_time = time.time() - round_start_time
                avg_epoch_loss = epoch_loss_total / total_seen_all_clients
                current_epoch = num_epochs

                logger.log_epoch(
                    epoch=current_epoch,
                    epoch_loss=avg_epoch_loss,
                    epoch_time=epoch_time,
                    world_size=num_clients,
                    extras={'train_acc': train_acc, 'val_acc': val_acc, 'lr': scheduler.get_last_lr()[0]}
                )
                scheduler.step()


        averaged_weights = average_models(client_models)

        if args.use_global_momentum:
            updated_weights, momentum_buffer = global_momentum_update(global_model, averaged_weights, momentum_buffer)
            global_model.load_state_dict(updated_weights)
        elif args.use_slowmo:
            updated_weights, slowmo_buffer = slowmo_update(global_model, averaged_weights, slowmo_buffer)
            global_model.load_state_dict(updated_weights)
        else:
            global_model.load_state_dict(averaged_weights)

        torch.save(global_model.state_dict(), f'checkpoints/global_model_epoch{current_epoch}.pth')

        best_val_acc = max(best_val_acc, val_acc)
    
        metrics_files = []
        for client_id in range(num_clients):
            path = os.path.join('logs', f"{train_name}_metrics_client{client_id}.json")
            if os.path.exists(path):
                metrics_files.append(path)
            else:
                print(f"[WARNING] Metrics file not found for client {client_id}: {path}")
                
    if metrics_files:
        plotter = TrainingPlotter(log_dir='logs', train_name=train_name, metrics_files=metrics_files, world_size=num_clients)
        plotter.plot_all_per_rank()  
    else:
        print("[ERROR] No metrics files found. Cannot generate plots.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=int, default=5, help='Local SGD steps before sync')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.001, help = 'Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer')
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=150, help='Number of total training epochs')
    parser.add_argument('--use_global_momentum', action='store_true', help='Use Global Momentum (BMUF-style) update')
    parser.add_argument('--use_slowmo', action='store_true', help='Use SlowMo optimizer update')



    args = parser.parse_args()

    dataset_size = 50000
    epochs_central = args.epochs
    total_batches_central = (dataset_size / args.batch_size) * epochs_central
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    epochs_local = int(total_batches_central / (dataset_size / (args.batch_size * world_size)))

    train_distributed(args, epochs_override=epochs_local)