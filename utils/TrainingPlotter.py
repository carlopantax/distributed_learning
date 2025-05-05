import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np


class TrainingPlotter:
    """
    Class to generate and save plots from training metrics.
    Optimized for visualizing longer training runs with many epochs.
    """

    def __init__(self, log_dir='logs', train_name=None, is_main_process=False):
        """
        Initialize the plotter.

        Args:
            log_dir (str): Directory where logs are saved
            train_name (str): Training experiment name
            is_main_process (bool): Whether this process is the main process
        """
        self.is_main_process = is_main_process

        # Only the main process handles plotting
        if self.is_main_process:
            self.log_dir = log_dir
            os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

            # Set experiment name
            if train_name is None:
                # Try to find the latest metrics file
                metrics_files = [f for f in os.listdir(log_dir) if f.endswith('_metrics.json')]
                if metrics_files:
                    # Get the most recent file
                    train_name = metrics_files[-1].replace('_metrics.json', '')
                else:
                    train_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

            self.train_name = train_name
            self.metrics_file = os.path.join(log_dir, f"{train_name}_metrics.json")
            self.plots_dir = os.path.join(log_dir, 'plots')

    def _load_metrics(self):
        """Load metrics from JSON file."""
        if not os.path.exists(self.metrics_file):
            print(f"Metrics file {self.metrics_file} not found.")
            return None

        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def _get_epoch_range(self, num_epochs):
        """Creates a proper epoch range for x-axis ticks."""
        if num_epochs <= 20:
            return list(range(1, num_epochs + 1))
        elif num_epochs <= 50:
            return list(range(1, num_epochs + 1, 2))
        elif num_epochs <= 100:
            return list(range(1, num_epochs + 1, 5))
        else:
            return list(range(1, num_epochs + 1, 10))

    def plot_epoch_loss(self):
        """Plot the training loss over epochs."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'epoch_loss' not in metrics:
            print("No epoch loss data found in metrics.")
            return

        plt.figure(figsize=(12, 6))

        # Get epoch numbers (1-indexed for display)
        epochs = list(range(1, len(metrics['epoch_loss']) + 1))

        # Plot the epoch loss
        plt.plot(epochs, metrics['epoch_loss'], 'o-', linewidth=2, markersize=8,
                 color='blue', label='Loss per Epoch')

        # Mark minimum loss
        min_loss = min(metrics['epoch_loss'])
        min_epoch = metrics['epoch_loss'].index(min_loss) + 1
        plt.scatter([min_epoch], [min_loss], color='green', s=100, zorder=5,
                    label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')

        # Add smoothed trendline if we have enough epochs
        if len(epochs) > 5:
            # Simple moving average for trend visualization
            window_size = min(5, len(epochs) // 5)
            if window_size > 1:
                smoothed = np.convolve(metrics['epoch_loss'],
                                       np.ones(window_size) / window_size,
                                       mode='valid')
                plt.plot(range(window_size, len(epochs) + 1), smoothed,
                         color='red', linestyle='--', linewidth=1.5,
                         label=f'{window_size}-Epoch Moving Avg')

        plt.title(f'Loss per Epoch - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Set appropriate x-ticks based on number of epochs
        plt.xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_epoch_loss.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Epoch loss plot saved to {save_path}")

    def plot_epoch_time(self):
        """Plot the time taken for each epoch."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'epoch_time' not in metrics:
            print("No epoch time data found in metrics.")
            return

        plt.figure(figsize=(12, 6))

        # Create x-axis labels (epoch numbers)
        epochs = list(range(1, len(metrics['epoch_time']) + 1))

        plt.bar(epochs, metrics['epoch_time'], color='skyblue')
        avg_time = np.mean(metrics['epoch_time'])
        plt.axhline(y=avg_time, color='r', linestyle='-',
                    label=f'Average: {avg_time:.2f}s')

        plt.title(f'Time per Epoch - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set appropriate x-ticks based on number of epochs
        plt.xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_epoch_times.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Epoch times plot saved to {save_path}")

    def plot_epoch_throughput(self):
        """Plot the throughput for each epoch."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'epoch_time' not in metrics:
            print("No epoch time data found for throughput calculation.")
            return

        plt.figure(figsize=(12, 6))

        # Create x-axis labels (epoch numbers)
        epochs = list(range(1, len(metrics['epoch_time']) + 1))

        # Calculate throughput as inverse of time (higher is better)
        # Note: This is a relative measure unless we know exact batch counts
        throughputs = [1.0 / t for t in metrics['epoch_time']]

        plt.bar(epochs, throughputs, color='lightgreen')
        avg_throughput = np.mean(throughputs)
        plt.axhline(y=avg_throughput, color='r', linestyle='-',
                    label=f'Average: {avg_throughput:.4f}')

        plt.title(f'Throughput per Epoch - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Throughput (1/seconds)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set appropriate x-ticks based on number of epochs
        plt.xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_epoch_throughput.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Epoch throughput plot saved to {save_path}")

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

        plt.figure(figsize=(12, 6))
        epochs = list(range(1, len(metrics['epoch_loss']) + 1))

        if has_train_acc:
            plt.plot(epochs, metrics['epoch_train_acc'], 'o-', color='blue',
                     label='Training Accuracy')

        if has_val_acc:
            plt.plot(epochs, metrics['epoch_val_acc'], 'o-', color='red',
                     label='Validation Accuracy')

            # Mark best validation accuracy
            if len(metrics['epoch_val_acc']) > 0:
                best_acc = max(metrics['epoch_val_acc'])
                best_epoch = metrics['epoch_val_acc'].index(best_acc) + 1
                plt.scatter([best_epoch], [best_acc], color='green', s=100, zorder=5,
                            label=f'Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})')

        plt.title(f'Model Accuracy - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set appropriate x-ticks based on number of epochs
        plt.xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_accuracy.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Accuracy plot saved to {save_path}")

    def plot_learning_rate(self):
        """Plot the learning rate over epochs if available."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'epoch_lr' not in metrics:
            print("No learning rate data found in metrics.")
            return

        plt.figure(figsize=(12, 6))

        epochs = list(range(1, len(metrics['epoch_lr']) + 1))
        plt.plot(epochs, metrics['epoch_lr'], 'o-', color='purple', linewidth=2)

        plt.title(f'Learning Rate Schedule - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Use log scale if learning rate changes significantly
        lr_max = max(metrics['epoch_lr'])
        lr_min = min(metrics['epoch_lr'])
        if lr_max > lr_min * 10:  # If max LR is more than 10x min LR
            plt.yscale('log')

        # Set appropriate x-ticks based on number of epochs
        plt.xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_learning_rate.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Learning rate plot saved to {save_path}")

    def plot_metric_comparison(self):
        """Plot multiple metrics on the same graph for comparison."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics or 'epoch_loss' not in metrics:
            print("No epoch metrics found for comparison.")
            return

        # Find all epoch-level metrics
        epoch_metrics = [key for key in metrics.keys() if key.startswith('epoch_') and key != 'epoch_time']

        if len(epoch_metrics) < 2:
            print("Not enough epoch metrics for comparison.")
            return

        plt.figure(figsize=(12, 6))
        epochs = list(range(1, len(metrics['epoch_loss']) + 1))

        # Use two different y-axes if we have various metrics with different scales
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # First metric on left y-axis (usually loss)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax1.plot(epochs, metrics['epoch_loss'], 'o-', color='blue', label='Loss')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Second metric on right y-axis (usually accuracy)
        has_second_axis = False
        for metric in epoch_metrics:
            if metric != 'epoch_loss':
                if not has_second_axis:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Other Metrics', color='red')
                    has_second_axis = True

                label = metric.replace('epoch_', '')
                ax2.plot(epochs, metrics[metric], 'o-', color='red' if metric == 'epoch_val_acc' else 'green',
                         label=label.capitalize())

        if has_second_axis:
            ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()
        plt.title(f'Training Metrics Comparison - {self.train_name}')

        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if has_second_axis:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')

        # Set appropriate x-ticks based on number of epochs
        ax1.set_xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_metrics_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Metrics comparison plot saved to {save_path}")

    def plot_summary(self):
        """Generate a summary plot with multiple subplots for key metrics."""
        if not self.is_main_process:
            return

        metrics = self._load_metrics()
        if not metrics:
            return

        # Count how many plots we'll need
        plot_count = 0
        has_loss = 'epoch_loss' in metrics
        has_time = 'epoch_time' in metrics
        has_train_acc = 'epoch_train_acc' in metrics
        has_val_acc = 'epoch_val_acc' in metrics
        has_lr = 'epoch_lr' in metrics

        if has_loss: plot_count += 1
        if has_time: plot_count += 1
        if has_train_acc or has_val_acc: plot_count += 1
        if has_lr: plot_count += 1

        if plot_count == 0:
            print("No plottable metrics found.")
            return

        # Configure the grid layout
        rows = min(3, plot_count)
        cols = (plot_count + rows - 1) // rows  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows * cols == 1:
            axes = np.array([axes])  # Make sure axes is always indexable as array
        axes = axes.flatten()

        current_ax = 0
        epochs = list(range(1, len(metrics['epoch_loss']) + 1)) if has_loss else []

        # Plot Loss
        if has_loss:
            ax = axes[current_ax]
            ax.plot(epochs, metrics['epoch_loss'], 'o-', color='blue')

            # Mark minimum loss
            min_loss = min(metrics['epoch_loss'])
            min_epoch = metrics['epoch_loss'].index(min_loss) + 1
            ax.scatter([min_epoch], [min_loss], color='green', s=50, zorder=5)

            ax.set_title('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            current_ax += 1

        # Plot Time per Epoch
        if has_time:
            ax = axes[current_ax]
            ax.bar(epochs, metrics['epoch_time'], color='skyblue')
            avg_time = np.mean(metrics['epoch_time'])
            ax.axhline(y=avg_time, color='r', linestyle='-', label=f'Avg: {avg_time:.2f}s')

            ax.set_title('Time per Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            ax.legend()
            current_ax += 1

        # Plot Accuracy
        if has_train_acc or has_val_acc:
            ax = axes[current_ax]

            if has_train_acc:
                ax.plot(epochs, metrics['epoch_train_acc'], 'o-', color='blue', label='Train')

            if has_val_acc:
                ax.plot(epochs, metrics['epoch_val_acc'], 'o-', color='red', label='Val')

                # Mark best validation accuracy
                if len(metrics['epoch_val_acc']) > 0:
                    best_acc = max(metrics['epoch_val_acc'])
                    best_epoch = metrics['epoch_val_acc'].index(best_acc) + 1
                    ax.scatter([best_epoch], [best_acc], color='green', s=50, zorder=5)

            ax.set_title('Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            current_ax += 1

        # Plot Learning Rate
        if has_lr:
            ax = axes[current_ax]
            ax.plot(epochs, metrics['epoch_lr'], 'o-', color='purple')

            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, linestyle='--', alpha=0.7)

            # Use log scale if learning rate changes significantly
            lr_max = max(metrics['epoch_lr'])
            lr_min = min(metrics['epoch_lr'])
            if lr_max > lr_min * 10:  # If max LR is more than 10x min LR
                ax.set_yscale('log')

            ax.set_xticks(self._get_epoch_range(len(epochs)))
            current_ax += 1

        # Hide unused subplots
        for i in range(current_ax, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.train_name}_summary.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Summary plot saved to {save_path}")

    def plot_all(self):
        """Generate all available plots."""
        if not self.is_main_process:
            return

        print(f"Generating all plots for {self.train_name}...")
        self.plot_epoch_loss()
        self.plot_epoch_time()
        self.plot_epoch_throughput()
        self.plot_accuracy()
        self.plot_learning_rate()
        self.plot_metric_comparison()
        self.plot_summary()
        print("All plots generated successfully!")