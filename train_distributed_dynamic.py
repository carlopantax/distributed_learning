import torch
import copy
import torch.nn as nn
import torch.optim as optim
from utils.TrainingLogger import TrainingLogger
from utils.data_utils import load_cifar100_for_clients as load_cifar100
from models.lenet import LeNet5
import time
from utils.outer_optimizer_bmuf import initialize_global_momentum, global_momentum_update
from utils.slowmo import initialize_slowmo_state, slowmo_update
import os
import math
from utils.dynamic_tau_controller import DynamicTauController
from utils.dynamic_tau_plotter import DynamicTauPlotter

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


def compute_gradient_norm(model):
    """Compute the L2 norm of gradients"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def average_models(models):
    """Average weights of multiple models."""
    avg_state_dict = copy.deepcopy(models[0].state_dict())
    for key in avg_state_dict:
        for i in range(1, len(models)):
            avg_state_dict[key] += models[i].state_dict()[key]
        avg_state_dict[key] = avg_state_dict[key] / len(models)
    return avg_state_dict


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = float(current_epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_distributed_dynamic_tau(args, epochs_override=None):
    """
    Enhanced training with dynamic tau adjustment per client
    """
    num_clients = args.world_size
    initial_tau = args.tau

    train_name = f"dynamic_tau_localSGD_lenet5_seq_cifar100_w{num_clients}_{time.strftime('%Y%m%d_%H%M%S')}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dynamic tau controller
    tau_controller = DynamicTauController(
        initial_tau=initial_tau,
        patience=args.patience,
        improvement_threshold=args.improvement_threshold
    )

    general_logger = TrainingLogger(
        log_dir='logs',
        train_name=train_name,
        client_id=100,  # Special ID for general logger
        num_clients=num_clients
    )

    general_logger.log(f"Training with Dynamic Tau: initial_tau={initial_tau}, min={args.min_tau}, max={args.max_tau}")
    general_logger.log(f"Device: {device}")
    general_logger.log(f"Config: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")

    # Dataset split
    client_loaders, client_val_loaders, testloader = load_cifar100(
        batch_size=args.batch_size,
        num_clients=num_clients
    )

    total_train_size = sum(len(loader.dataset) for loader in client_loaders)
    general_logger.log(f"[INFO] Total train set size: {total_train_size}")

    num_epochs = epochs_override if epochs_override is not None else args.epochs
    global_model = LeNet5(num_classes=100).to(device)

    # Ensure directories exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Initialize buffers for different optimizers
    momentum_buffer = None
    slowmo_buffer = None

    if args.use_global_momentum:
        momentum_buffer = initialize_global_momentum(global_model)
    elif args.use_slowmo:
        slowmo_buffer = initialize_slowmo_state(global_model)  # Fixed: proper function call

    # Initialize sync control variables (moved outside of conditional blocks)
    clients_to_sync = set(range(num_clients))
    total_syncs = 0
    max_syncs = num_epochs  # Maximum number of sync rounds

    while total_syncs < max_syncs and len(clients_to_sync) > 0:
        sync_start_time = time.time()
        general_logger.log(f"--- Sync Round {total_syncs + 1} ---")
        general_logger.log(f"Clients to sync: {list(clients_to_sync)}")

        models_to_average = []
        next_round_clients = set()

        for client_idx in clients_to_sync:
            # Initialize client model
            model = LeNet5(num_classes=100).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=num_epochs,
                                                        base_lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            logger = TrainingLogger(
                log_dir='logs',
                train_name=train_name,
                client_id=client_idx,
                num_clients=num_clients
            )

            current_tau = tau_controller.get_client_tau(client_idx)
            general_logger.log(f"Client {client_idx}: Starting with tau={current_tau}")

            # Initialize variables for early sync detection
            should_sync = False
            new_tau = current_tau
            grad_norm = 0.0

            # Train until tau or early stopping
            for local_epoch in range(current_tau):
                model.train()
                batch_start_time = time.time()
                running_correct = 0
                total_seen = 0
                running_loss = 0.0
                total_batches = len(client_loaders[client_idx])
                log_freq = 10  # Log every 10 batches

                for batch_idx, (inputs, labels) in enumerate(client_loaders[client_idx], start=1):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    # Compute gradient norm before optimizer step
                    grad_norm = compute_gradient_norm(model)

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

                        logger.log_metrics(
                            epoch=tau_controller.client_states[client_idx]['total_epochs'],
                            batch_idx=batch_idx - 1,  # 0-indexed for logger
                            loss=avg_loss,
                            batch_size=inputs.size(0),
                            train_size=len(client_loaders[client_idx].dataset),
                            extras={
                                'train_acc': train_acc,
                                'lr': scheduler.get_last_lr()[0],
                                'grad_norm': grad_norm,
                                'current_tau': current_tau,
                                'local_epoch': local_epoch + 1
                            }
                        )

                # Epoch-level metrics
                epoch_loss = running_loss / total_seen
                train_acc = 100.0 * running_correct / total_seen
                val_acc = compute_accuracy(model, client_val_loaders[client_idx], device)
                model_divergence = tau_controller.compute_model_divergence(model, global_model)

                tau_controller.step_epoch(client_idx)

                # Check if should sync early
                should_sync, new_tau = tau_controller.should_sync_early(
                    client_id=client_idx,
                    current_val_acc=val_acc,
                    current_loss=epoch_loss,
                    gradient_norm=grad_norm,
                    model_divergence=model_divergence   
                )


                logger.log(
                    f"Client {client_idx} | Local Epoch {local_epoch + 1}/{current_tau} | "
                    f"Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}% | Grad Norm: {grad_norm:.4f} | "
                    f"Divergence: {model_divergence:.4f} | "
                    f"Should Sync: {should_sync} | New Tau: {new_tau}"
                )


                # Log epoch using your logger
                current_global_epoch = tau_controller.client_states[client_idx]['total_epochs']
                epoch_time = time.time() - sync_start_time
                logger.log_epoch(
                    epoch=current_global_epoch,
                    epoch_loss=epoch_loss,
                    epoch_time=epoch_time,
                    world_size=num_clients,
                    extras={
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'lr': scheduler.get_last_lr()[0],
                        'current_tau': current_tau,
                        'grad_norm': grad_norm,
                        'model_divergence': model_divergence,
                        'local_epoch': local_epoch + 1,
                        'sync_round': total_syncs + 1
                    }
                )

                if should_sync:
                    # Log tau change if different
                    if new_tau != current_tau:
                        reason = tau_controller.get_tau_change_reason(client_idx)
                        general_logger.log(
                            f"Client {client_idx}: Tau changed from {current_tau} to {new_tau} "
                            f"(Reason: {reason})"
                        )

                    general_logger.log(f"Client {client_idx}: Early sync at epoch {local_epoch + 1}, new tau={new_tau}")
                    break

                scheduler.step()

            # Add model for averaging and prepare for next round
            models_to_average.append(model)
            tau_controller.on_sync(client_idx, new_tau)

            # Check if client should continue training
            if tau_controller.client_states[client_idx]['total_epochs'] < num_epochs:
                next_round_clients.add(client_idx)

        # Average models and update global model
        if models_to_average:
            averaged_weights = average_models(models_to_average)

            if args.use_global_momentum:
                updated_weights, momentum_buffer = global_momentum_update(global_model, averaged_weights,
                                                                          momentum_buffer)
                global_model.load_state_dict(updated_weights)
            elif args.use_slowmo:
                updated_weights, slowmo_buffer = slowmo_update(global_model, averaged_weights, slowmo_buffer)
                global_model.load_state_dict(updated_weights)
            else:
                global_model.load_state_dict(averaged_weights)

            # Ensure checkpoint directory exists before saving
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/global_model_sync{total_syncs + 1}.pth'
            torch.save(global_model.state_dict(), checkpoint_path)
            general_logger.log(f"Saved global model checkpoint: {checkpoint_path}")

        clients_to_sync = next_round_clients
        total_syncs += 1

        general_logger.log(f"Sync round {total_syncs} completed in {time.time() - sync_start_time:.2f}s")

    # Final evaluation
    final_test_acc = compute_accuracy(global_model, testloader, device)
    general_logger.log(f"Final test accuracy: {final_test_acc:.2f}%")

    # Log tau statistics
    for client_id in range(num_clients):
        if client_id in tau_controller.client_states:
            stats = tau_controller.get_client_stats(client_id)
            general_logger.log(
                f"Client {client_id} final stats: "
                f"Total epochs: {stats['total_epochs']}, "
                f"Syncs: {stats['sync_count']}, "
                f"Final tau: {stats['current_tau']}, "
                f"Best val acc: {stats['best_val_acc']:.2f}, "
                f"Last reason: {stats['last_reason']}"
            )

    # Generate plots using enhanced dynamic tau plotter
    general_logger.log("Generating training plots...")

    # Find all metrics files for this training run
    metrics_files = []
    for client_id in range(num_clients):
        path = os.path.join('logs', f"{train_name}_metrics_client{client_id}.json")
        if os.path.exists(path):
            metrics_files.append(path)
        else:
            general_logger.log(f"[WARNING] Metrics file not found for client {client_id}: {path}")

    # Also include general logger metrics
    general_metrics_path = os.path.join('logs', f"{train_name}_metrics_client100.json")
    if os.path.exists(general_metrics_path):
        metrics_files.append(general_metrics_path)

    if metrics_files:
        try:
            plotter = DynamicTauPlotter(
                log_dir='logs',
                train_name=train_name,
                metrics_files=metrics_files,
                world_size=num_clients
            )
            plotter.plot_all_dynamic_tau()
            general_logger.log("All plots including dynamic tau analysis generated successfully!")
        except Exception as e:
            general_logger.log(f"[WARNING] Could not generate plots: {str(e)}")
    else:
        general_logger.log("[ERROR] No metrics files found. Cannot generate plots.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=int, default=16, help='Initial tau (local SGD steps before sync)')
    parser.add_argument('--min_tau', type=int, default=4, help='Minimum tau value')
    parser.add_argument('--max_tau', type=int, default=32, help='Maximum tau value')
    parser.add_argument('--patience', type=int, default=3, help='Patience for tau adjustment')
    parser.add_argument('--improvement_threshold', type=float, default=0.01, help='Threshold for improvement detection')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
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

    train_distributed_dynamic_tau(args, epochs_override=epochs_local)