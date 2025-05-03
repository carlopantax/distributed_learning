import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np


class TrainingPlotter:
    """
    Class to generate and save plots from training metrics.
    """

    def __init__(self, log_dir='logs', train_name=None, is_main_process=False):
        """
        Initializes the plotter.

        Args:
            log_dir (str): Directory where logs are saved
            train_name (str): Experiment name
            is_main_process (bool): Whether this process is the main process
        """
        self.is_main_process = is_main_process

        if self.is_main_process:
            self.log_dir = log_dir
            os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

            if train_name is None:
                # Try to find the latest metrics file
                metrics_files = [f for f in os.listdir(log_dir) if f.endswith('_metrics.json')]
                if metrics_files:
                    # Get the most recent file
                    train_name = metrics_files[-1].replace('_metrics.json', '')
                else:
                    train_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

            self.exp_name = train_name
            self.metrics_file = os.path.join(log_dir, f"{train_name}_metrics.json")
            self.plots_dir = os.path.join(log_dir, 'plots')

    def _load_metrics(self):
        """Loads metrics from JSON file."""
        if not os.path.exists(self.metrics_file):
            print(f"Metrics file {self.metrics_file} not found.")
            return None

        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def plot_training_loss(self):
        """Plots the training loss over time."""
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
        """Plots the training throughput over time."""
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
        """Plots the time taken for each epoch."""
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

    def plot_all(self):
        """Generates all available plots."""
        if not self.is_main_process:
            return

        self.plot_training_loss()
        self.plot_throughput()
        self.plot_epoch_times()

        # Create a summary plot with multiple subplots
        metrics = self._load_metrics()
        if not metrics:
            return

        plt.figure(figsize=(15, 10))

        # Plot loss
        if 'loss' in metrics:
            plt.subplot(2, 2, 1)
            plt.plot(metrics['loss'], label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot throughput
        if 'imgs_per_sec' in metrics:
            plt.subplot(2, 2, 2)
            plt.plot(metrics['imgs_per_sec'], label='Images/second')
            plt.title('Training Throughput')
            plt.xlabel('Batch')
            plt.ylabel('Images/second')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot epoch times
        if 'epoch_time' in metrics:
            plt.subplot(2, 2, 3)
            epochs = list(range(1, len(metrics['epoch_time']) + 1))
            plt.bar(epochs, metrics['epoch_time'], color='skyblue')
            plt.title('Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Plot epoch loss if available
        if 'epoch_loss' in metrics:
            plt.subplot(2, 2, 4)
            epochs = list(range(1, len(metrics['epoch_loss']) + 1))
            plt.plot(epochs, metrics['epoch_loss'], 'o-', color='green')
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.exp_name}_summary.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Summary plot saved to {save_path}")