import os
import json
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class TrainingLogger:
    """
    Class to manage logging during training in a distributed setting.
    Handles log rotation, formatting, and saving metrics to file.
    """

    def __init__(self, log_dir='logs', exp_name=None, rank=0, is_main_process=False):
        """
        Initialize the logger.

        Args:
            log_dir (str): Directory to save logs
            exp_name (str): Experiment name for the log files
            rank (int): Process rank in distributed training
            is_main_process (bool): Whether this process is the main process
        """
        self.rank = rank
        self.is_main_process = is_main_process
        self.metrics = defaultdict(list)
        self.start_time = time.time()

        # Only the main process saves logs to avoid conflicts
        if self.is_main_process:
            os.makedirs(log_dir, exist_ok=True)

            # Create experiment name based on timestamp if not provided
            if exp_name is None:
                exp_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

            self.exp_name = exp_name
            self.log_dir = log_dir
            self.log_file = os.path.join(log_dir, f"{exp_name}.log")
            self.metrics_file = os.path.join(log_dir, f"{exp_name}_metrics.json")

            # Configure logging
            self.logger = logging.getLogger(f"trainer_{rank}")
            self.logger.setLevel(logging.INFO)

            # Create file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            self.logger.info(f"Started training experiment: {exp_name}")
            self.logger.info(f"Log file: {self.log_file}")

    def log(self, message, level='info'):
        """Log a message with the specified level."""
        if self.is_main_process:
            if level.lower() == 'info':
                self.logger.info(message)
            elif level.lower() == 'warning':
                self.logger.warning(message)
            elif level.lower() == 'error':
                self.logger.error(message)
            elif level.lower() == 'debug':
                self.logger.debug(message)

    def log_metrics(self, epoch, batch_idx, loss, batch_size, train_size, extras=None):
        """
        Log training metrics for the current batch.

        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            loss (float): Current loss value
            batch_size (int): Size of the batch
            train_size (int): Total size of the training set
            extras (dict): Extra metrics to log
        """
        if not self.is_main_process:
            return

        elapsed = time.time() - self.start_time
        progress = 100. * batch_idx * batch_size / train_size
        imgs_per_sec = batch_idx * batch_size / elapsed if elapsed > 0 else 0

        # Store metrics
        self.metrics['epoch'].append(epoch)
        self.metrics['batch'].append(batch_idx)
        self.metrics['loss'].append(loss)
        self.metrics['imgs_per_sec'].append(imgs_per_sec)
        self.metrics['progress'].append(progress)
        self.metrics['elapsed'].append(elapsed)

        if extras:
            for key, value in extras.items():
                self.metrics[key].append(value)

        log_msg = (
            f"Epoch: {epoch + 1} | "
            f"Batch: {batch_idx + 1}/{train_size // batch_size} | "
            f"Loss: {loss:.4f} | "
            f"Images/sec: {imgs_per_sec:.2f} | "
            f"Progress: {progress:.1f}% | "
            f"Time: {elapsed:.2f}s"
        )

        # Add extras to log message if provided
        if extras:
            extras_str = " | ".join([f"{k}: {v:.4f}" for k, v in extras.items()])
            log_msg += f" | {extras_str}"

        self.log(log_msg)

    def log_epoch(self, epoch, epoch_loss, epoch_time, world_size=1, extras=None):
        """
        Log metrics for completed epoch.

        Args:
            epoch (int): Current epoch
            epoch_loss (float): Average loss for the epoch
            epoch_time (float): Time taken for the epoch
            world_size (int): Number of processes in distributed training
            extras (dict): Extra metrics to log
        """
        if not self.is_main_process:
            return

        # Store epoch metrics
        self.metrics['epoch_loss'].append(epoch_loss)
        self.metrics['epoch_time'].append(epoch_time)

        if extras:
            for key, value in extras.items():
                self.metrics[f"epoch_{key}"].append(value)

        log_msg = (
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s | "
            f"Average Loss: {epoch_loss:.4f}"
        )

        # Add extras to log message if provided
        if extras:
            extras_str = " | ".join([f"{k}: {v:.4f}" for k, v in extras.items()])
            log_msg += f" | {extras_str}"

        self.log(log_msg)

    def log_training_complete(self, total_time, total_images, world_size=1):
        """
        Log final training statistics.

        Args:
            total_time (float): Total training time
            total_images (int): Total number of images processed
            world_size (int): Number of processes in distributed training
        """
        if not self.is_main_process:
            return

        throughput = total_images / total_time

        self.metrics['total_time'] = total_time
        self.metrics['throughput'] = throughput
        self.metrics['world_size'] = world_size

        self.log(f"Training completed in {total_time:.2f} seconds.")
        self.log(f"Average throughput: {throughput:.2f} Images/sec")
        self.log(f"Total processes: {world_size}")

        # Save all metrics to JSON file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.log(f"Metrics saved to {self.metrics_file}")


class TrainingPlotter:
    """
    Class to generate and save plots from training metrics.
    """

    def __init__(self, log_dir='logs', exp_name=None, is_main_process=False):
        """
        Initialize the plotter.

        Args:
            log_dir (str): Directory where logs are saved
            exp_name (str): Experiment name
            is_main_process (bool): Whether this process is the main process
        """
        self.is_main_process = is_main_process

        # Only the main process handles plotting
        if self.is_main_process:
            self.log_dir = log_dir
            os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

            # Set experiment name
            if exp_name is None:
                # Try to find the latest metrics file
                metrics_files = [f for f in os.listdir(log_dir) if f.endswith('_metrics.json')]
                if metrics_files:
                    # Get the most recent file
                    exp_name = metrics_files[-1].replace('_metrics.json', '')
                else:
                    exp_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

            self.exp_name = exp_name
            self.metrics_file = os.path.join(log_dir, f"{exp_name}_metrics.json")
            self.plots_dir = os.path.join(log_dir, 'plots')

    def _load_metrics(self):
        """Load metrics from JSON file."""
        if not os.path.exists(self.metrics_file):
            print(f"Metrics file {self.metrics_file} not found.")
            return None

        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def plot_training_loss(self):
        """Plot the training loss over time."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'loss' not in metrics:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(metrics['loss'], label='Training Loss')

        # If we have epoch loss data, add it as points
        if 'epoch_loss' in metrics and 'epoch' in metrics:
            # Find the last batch index for each epoch
            unique_epochs = sorted(set(metrics['epoch']))
            epoch_indices = []
            for e in unique_epochs:
                indices = [i for i, epoch in enumerate(metrics['epoch']) if epoch == e]
                if indices:
                    epoch_indices.append(max(indices))

            # Extract loss values at those indices
            epoch_x = [i for i in range(len(metrics['epoch_loss']))]
            epoch_y = metrics['epoch_loss']

            plt.scatter(epoch_x, epoch_y, color='red', s=50, label='Epoch Average')

        plt.title(f'Training Loss Over Time - {self.exp_name}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        save_path = os.path.join(self.plots_dir, f"{self.exp_name}_loss.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Loss plot saved to {save_path}")

    def plot_throughput(self):
        """Plot the training throughput over time."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'imgs_per_sec' not in metrics:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(metrics['imgs_per_sec'], label='Images/second')

        if 'throughput' in metrics:
            plt.axhline(y=metrics['throughput'], color='r', linestyle='-',
                        label=f'Average: {metrics["throughput"]:.2f} imgs/sec')

        plt.title(f'Training Throughput - {self.exp_name}')
        plt.xlabel('Batch')
        plt.ylabel('Images/second')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        save_path = os.path.join(self.plots_dir, f"{self.exp_name}_throughput.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Throughput plot saved to {save_path}")

    def plot_epoch_times(self):
        """Plot the time taken for each epoch."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'epoch_time' not in metrics:
            return

        plt.figure(figsize=(10, 6))

        # Create x-axis labels (epoch numbers)
        epochs = list(range(1, len(metrics['epoch_time']) + 1))

        plt.bar(epochs, metrics['epoch_time'], color='skyblue')
        plt.axhline(y=np.mean(metrics['epoch_time']), color='r', linestyle='-',
                    label=f'Average: {np.mean(metrics["epoch_time"]):.2f}s')

        plt.title(f'Time per Epoch - {self.exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(epochs)

        save_path = os.path.join(self.plots_dir, f"{self.exp_name}_epoch_times.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Epoch times plot saved to {save_path}")

    def plot_accuracy(self):
        """Plot the training and validation accuracy over epochs."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics:
            return

        # Check if we have accuracy metrics
        has_train_acc = 'epoch_train_acc' in metrics
        has_val_acc = 'epoch_val_acc' in metrics

        if not (has_train_acc or has_val_acc):
            print("No accuracy data found in metrics.")
            return

        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(metrics['epoch_loss']) + 1))

        if has_train_acc:
            plt.plot(epochs, metrics['epoch_train_acc'], 'o-', color='blue', label='Training Accuracy')

        if has_val_acc:
            plt.plot(epochs, metrics['epoch_val_acc'], 'o-', color='red', label='Validation Accuracy')

        plt.title(f'Model Accuracy - {self.exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        save_path = os.path.join(self.plots_dir, f"{self.exp_name}_accuracy.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Accuracy plot saved to {save_path}")

    def plot_all(self):
        """Generate all available plots."""
        if not self.is_main_process:
            return

        self.plot_training_loss()
        self.plot_throughput()
        self.plot_epoch_times()
        self.plot_accuracy()  # Add the new accuracy plot

        # Create a summary plot with multiple subplots
        metrics = self._load_metrics()
        if not metrics:
            return

        plt.figure(figsize=(15, 12))  # Increased height for more subplots

        # Plot loss
        if 'loss' in metrics:
            plt.subplot(3, 2, 1)  # Changed to 3x2 grid
            plt.plot(metrics['loss'], label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot throughput
        if 'imgs_per_sec' in metrics:
            plt.subplot(3, 2, 2)
            plt.plot(metrics['imgs_per_sec'], label='Images/second')
            plt.title('Training Throughput')
            plt.xlabel('Batch')
            plt.ylabel('Images/second')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot epoch times
        if 'epoch_time' in metrics:
            plt.subplot(3, 2, 3)
            epochs = list(range(1, len(metrics['epoch_time']) + 1))
            plt.bar(epochs, metrics['epoch_time'], color='skyblue')
            plt.title('Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot epoch loss if available
        if 'epoch_loss' in metrics:
            plt.subplot(3, 2, 4)
            epochs = list(range(1, len(metrics['epoch_loss']) + 1))
            plt.plot(epochs, metrics['epoch_loss'], 'o-', color='green')
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot accuracy if available
        has_train_acc = 'epoch_train_acc' in metrics
        has_val_acc = 'epoch_val_acc' in metrics

        if has_train_acc or has_val_acc:
            plt.subplot(3, 2, 5)
            epochs = list(range(1, len(metrics['epoch_loss']) + 1))

            if has_train_acc:
                plt.plot(epochs, metrics['epoch_train_acc'], 'o-', color='blue', label='Train')

            if has_val_acc:
                plt.plot(epochs, metrics['epoch_val_acc'], 'o-', color='red', label='Val')

            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot learning rate if available
        if 'epoch_lr' in metrics:
            plt.subplot(3, 2, 6)
            epochs = list(range(1, len(metrics['epoch_lr']) + 1))
            plt.plot(epochs, metrics['epoch_lr'], 'o-', color='purple')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.exp_name}_summary.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Summary plot saved to {save_path}")