import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob
import re



class TrainingPlotter:
    """
    Utilities to visualize and save figures from training metrics.

    Output layout (under `<log_dir>/plots`):
      - {train_name}_client{rank}_epoch_loss.png
      - {train_name}_client{rank}_accuracy.png
      - {train_name}_epoch_times.png
      - {train_name}_epoch_throughput.png
      - {train_name}_accuracy.png
      - {train_name}_learning_rate.png
      - {train_name}_metrics_comparison.png
      - {train_name}_summary.png
      - Per-rank variants for comparison/summary where applicable
    """

    def __init__(self, log_dir='logs', train_name=None, metrics_files=None, world_size=1):
        """
        Prepare the plotter and resolve the experiment name and metrics files.

        Args:
            log_dir: Directory containing logs and metrics.
            train_name: Run identifier.
            metrics_files: Explicit list of metrics files to use.
            world_size: Number of clients/workers expected.
        """

        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        self.plots_dir = os.path.join(log_dir, 'plots')

        if train_name is None:
            metrics_files = [f for f in os.listdir(log_dir) if f.endswith('_metrics.json') or f.endswith('_metrics_combined.json')]
            metrics_files.sort() 
            if metrics_files:
                train_name = metrics_files[-1].replace('_metrics.json', '').replace('_metrics_combined.json', '')
            else:
                train_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.train_name = train_name
        self.world_size = world_size

        if metrics_files is not None:
                self.metrics_files = metrics_files
        else:
            centralized_path = os.path.join(log_dir, f"{self.train_name}_metrics.json")
            distributed_path = os.path.join(log_dir, f"{self.train_name}_metrics_combined.json")

            if os.path.exists(centralized_path):
                self.metrics_files = [centralized_path]
            elif os.path.exists(distributed_path):
                self.metrics_files = [distributed_path]
                print(f"[INFO] Falling back to combined metrics: {os.path.basename(distributed_path)}")
            else:
                print(f"[WARNING] No metrics file found for {self.train_name}.")
                self.metrics_files = []
        
    
        
    def _load_all_metrics(self):
        """
        Load all metrics files for this training session from all clients and rounds.
        Returns a list of (filename, data) tuples.
        """
        metrics_data = []

        for rank in range(self.world_size):
            
            pattern = os.path.join(self.log_dir, f"{self.train_name}_metrics_client{rank}*.json")
            files = sorted(glob.glob(pattern))

            if not files:
                print(f"[WARNING] No metrics file found for client {rank}")
                continue

            for file in files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        metrics_data.append((file, data))
                except Exception as e:
                    print(f"[ERROR] Could not load {file}: {e}")

        return metrics_data  # [(filename, dict), ...]


    def _group_metrics_by_rank(self, all_metrics):
        """
        Merge metrics across rounds for each client.

        Args:
            all_metrics: Output of `_load_all_metrics()`.

        Returns:
            Dict mapping client_id -> merged metrics dict (lists concatenated).
        """
        rank_metrics = defaultdict(lambda: defaultdict(list))

        for filename, metrics in all_metrics:
            match = re.search(r'_client(\d+)', filename)
            if not match:
                continue
            rank = int(match.group(1))

            for key, value in metrics.items():
                if isinstance(value, list):
                    rank_metrics[rank][key].extend(value)
                else:
                    rank_metrics[rank][key] = value
        return {rank: dict(metrics) for rank, metrics in rank_metrics.items()}


    def _convert_metrics_list_to_dict(self, metrics_list):
        """
        Convert a list of metrics dicts into a single dict of lists.

        Args:
            metrics_list: Iterable of per-epoch/round dicts.

        Returns:
            Merged dict where each key maps to a list of values.
        """
        if not isinstance(metrics_list, list) or not metrics_list:
            return {}

        merged = defaultdict(list)
        for entry in metrics_list:
            for key, value in entry.items():
                if isinstance(value, list):
                    merged[key].extend(value)
                else:
                    merged[key].append(value)
        return dict(merged)


    def _get_epoch_range(self, num_epochs):
        """
        Choose readable tick spacing for the x-axis given the epoch count.

        Args:
            num_epochs: Number of epochs available.

        Returns:
            A list of epoch indices to use as ticks.
        """
        if num_epochs <= 20:
            return list(range(1, num_epochs + 1))
        elif num_epochs <= 50:
            return list(range(1, num_epochs + 1, 2))
        elif num_epochs <= 100:
            return list(range(1, num_epochs + 1, 5))
        else:
            return list(range(1, num_epochs + 1, 10))

    def plot_epoch_loss(self):
        """
        Plot loss per epoch for each client (with min-loss marker and optional smoothing).
        """
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        for client_id, metrics in metrics_by_rank.items():
            epoch_loss = metrics.get('epoch_loss', [])
            if not epoch_loss:
                print(f"No epoch_loss for client {client_id}")
                continue

            epochs = list(range(1, len(epoch_loss) + 1))

            plt.figure(figsize=(12, 6))
            plt.plot(epochs, epoch_loss, 'o-', linewidth=2, markersize=8,
                    color='blue', label='Loss per Epoch')

            min_loss = min(epoch_loss)
            min_epoch = epoch_loss.index(min_loss) + 1
            plt.scatter([min_epoch], [min_loss], color='green', s=100, zorder=5,
                        label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')

            if len(epochs) > 5:
                window_size = min(5, len(epochs) // 5)
                if window_size > 1:
                    smoothed = np.convolve(epoch_loss, np.ones(window_size) / window_size, mode='valid')
                    plt.plot(range(window_size, len(epochs) + 1), smoothed,
                            color='red', linestyle='--', linewidth=1.5,
                            label=f'{window_size}-Epoch Moving Avg')

            plt.title(f'Loss per Epoch - Client {client_id}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.xticks(self._get_epoch_range(len(epochs)))

            save_path = os.path.join(self.plots_dir, f"{self.train_name}_client{client_id}_epoch_loss.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[Plot] Saved epoch loss for client {client_id} to {save_path}")


    def plot_epoch_time(self):
        """
        Plot the average epoch time across clients (aligned, padded if needed).
        """
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        all_epoch_times = []
        for client_id, metrics in metrics_by_rank.items():
            epoch_times = metrics.get('epoch_time', [])
            if epoch_times:
                all_epoch_times.append(epoch_times)

        if not all_epoch_times:
            print("No epoch_time data found for any client.")
            return

        num_epochs = max(len(seq) for seq in all_epoch_times)

        def pad_or_truncate(seq, length):
            return seq[:length] + [0.0] * (length - len(seq))

        avg_times = np.mean([pad_or_truncate(seq, num_epochs) for seq in all_epoch_times], axis=0)

        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(12, 6))
        plt.bar(epochs, avg_times, color='skyblue')
        avg_time = np.mean(avg_times)
        plt.axhline(y=avg_time, color='red', linestyle='-', label=f'Average: {avg_time:.2f}s')

        plt.title(f'Time per Epoch - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(self._get_epoch_range(num_epochs))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_epoch_times.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Epoch time plot saved to {save_path}")


    def plot_epoch_throughput(self):
        """
        Plot a simple relative throughput proxy per epoch (1 / avg epoch time).
        """
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        all_epoch_times = []
        for client_id, metrics in metrics_by_rank.items():
            epoch_times = metrics.get('epoch_time', [])
            if epoch_times:
                all_epoch_times.append(epoch_times)

        if not all_epoch_times:
            print("No epoch_time data found.")
            return

        num_epochs = max(len(seq) for seq in all_epoch_times)

        def pad_or_truncate(seq, length):
            return seq[:length] + [0.0] * (length - len(seq))

        avg_times = np.mean([pad_or_truncate(seq, num_epochs) for seq in all_epoch_times], axis=0)
        throughputs = [1.0 / t if t > 0 else 0 for t in avg_times]

        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(12, 6))
        plt.bar(epochs, throughputs, color='lightgreen')
        avg_throughput = np.mean(throughputs)
        plt.axhline(y=avg_throughput, color='red', linestyle='-', label=f'Average: {avg_throughput:.4f}')

        plt.title(f'Throughput per Epoch - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Throughput (1/seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(self._get_epoch_range(num_epochs))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_epoch_throughput.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Epoch throughput plot saved to {save_path}")

    
    def plot_accuracy_per_client(self):
        """Plot training and validation accuracy for each client individually, highlighting best val accuracy."""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        for client_id, metrics in metrics_by_rank.items():
            train_acc = metrics.get('epoch_train_acc', [])
            val_acc = metrics.get('epoch_val_acc', [])
            if not train_acc and not val_acc:
                continue

            num_epochs = max(len(train_acc), len(val_acc))
            epochs = list(range(1, num_epochs + 1))

            def pad_or_truncate(seq, length):
                return seq[:length] + [0.0] * (length - len(seq))

            train_acc = pad_or_truncate(train_acc, num_epochs)
            val_acc = pad_or_truncate(val_acc, num_epochs)

            plt.figure(figsize=(12, 6))
            if train_acc:
                plt.plot(epochs, train_acc, 'o-', color='blue', label='Train Accuracy')
            if val_acc:
                plt.plot(epochs, val_acc, 'o-', color='red', label='Validation Accuracy')
                best_val = max(val_acc)
                best_epoch = val_acc.index(best_val) + 1
                plt.scatter([best_epoch], [best_val], color='green', s=100, zorder=5,
                            label=f'Best Val Acc: {best_val:.2f}% (Epoch {best_epoch})')

            plt.title(f"Client {client_id} Accuracy - {self.train_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.xticks(self._get_epoch_range(num_epochs))

            filename = f"{self.train_name}_client{client_id}_accuracy.png"
            save_path = os.path.join(self.plots_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"Accuracy plot for client {client_id} saved to {save_path}")



    def plot_accuracy(self):
        """Plot the training and validation accuracy over epochs (aggregated by client)."""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        train_acc_all = []
        val_acc_all = []

        for client_id, metrics in metrics_by_rank.items():
            train_acc = metrics.get('epoch_train_acc', [])
            val_acc = metrics.get('epoch_val_acc', [])
            if train_acc:
                train_acc_all.append(train_acc)
            if val_acc:
                val_acc_all.append(val_acc)

        if not train_acc_all and not val_acc_all:
            print("No accuracy data found.")
            return

        num_epochs = max(len(seq) for seq in train_acc_all + val_acc_all)
        epochs = list(range(1, num_epochs + 1))

        def pad_or_truncate(seq, length):
            return seq[:length] + [0.0] * (length - len(seq))

        train_avg = np.mean([pad_or_truncate(seq, num_epochs) for seq in train_acc_all], axis=0) if train_acc_all else []
        val_avg = np.mean([pad_or_truncate(seq, num_epochs) for seq in val_acc_all], axis=0) if val_acc_all else []

        plt.figure(figsize=(12, 6))
        if len(train_avg):
            plt.plot(epochs, train_avg, 'o-', color='blue', label='Train Accuracy (avg)')
        if len(val_avg):
            plt.plot(epochs, val_avg, 'o-', color='red', label='Validation Accuracy (avg)')
            best_acc = np.max(val_avg)
            best_epoch = np.argmax(val_avg) + 1
            plt.scatter([best_epoch], [best_acc], color='green', s=100, zorder=5,
                        label=f'Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})')

        plt.title(f'Model Accuracy - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(self._get_epoch_range(num_epochs))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_accuracy.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Accuracy plot saved to {save_path}")


    def plot_learning_rate(self):
        """Plot the learning rate over epochs (averaged across clients)."""

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)
        lr_all = []

        for metrics in metrics_by_rank.values():
            lr = metrics.get('epoch_lr', [])
            if lr:
                lr_all.append(lr)

        if not lr_all:
            print("No learning rate data found in any client.")
            return

        num_epochs = max(len(seq) for seq in lr_all)

        def pad_or_truncate(seq, length):
            return seq[:length] + [0.0] * (length - len(seq))

        epoch_lr = np.mean([pad_or_truncate(seq, num_epochs) for seq in lr_all], axis=0)

        epochs = list(range(1, len(epoch_lr) + 1))
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, epoch_lr, 'o-', color='purple', linewidth=2)

        plt.title(f'Learning Rate Schedule - {self.train_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)

        if max(epoch_lr) > min(epoch_lr) * 10:
            plt.yscale('log')

        plt.xticks(self._get_epoch_range(len(epoch_lr)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_learning_rate.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Learning rate plot saved to {save_path}")


    def plot_metric_comparison(self):
        """
        Overlay multiple epoch-level metrics for the run on a shared figure.

        Produces a dual-axis plot:
          - Left Y: losses
          - Right Y: other metrics (e.g., accuracy, LR)
        """

        all_metrics = self._load_all_metrics()

        if not all_metrics:
            print("No metrics found.")
            return

        # Convert list of dicts to dict of lists
        metrics_list = [m[1] for m in all_metrics if isinstance(m[1], dict)]
        metrics = self._convert_metrics_list_to_dict(metrics_list)

        if not metrics:
            print("No valid metrics found.")
            return

        
        epoch_metrics = [key for key in metrics.keys() if key.startswith('epoch_') and key != 'epoch_time']
        if not epoch_metrics:
            print("No epoch metrics found for comparison.")
            return

        keys = ['epoch_loss', 'epoch_val_loss', 'epoch_train_acc', 'epoch_val_acc', 'epoch_lr']
        available = [k for k in keys if k in metrics and isinstance(metrics[k], list)]
        if len(available) < 2:
            print("Not enough metrics for comparison.")
            return

        epochs = list(range(1, len(metrics[available[0]]) + 1))
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')

        if 'epoch_loss' in metrics:
            ax1.plot(epochs, metrics['epoch_loss'], 'o-', color='blue', label='Train Loss')
        if 'epoch_val_loss' in metrics:
            ax1.plot(epochs, metrics['epoch_val_loss'], 's--', color='cyan', label='Val Loss')

        ax1.tick_params(axis='y', labelcolor='blue')

        has_second_axis = False
        for key in available:
            if key not in ['epoch_loss', 'epoch_val_loss']:
                if not has_second_axis:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Other Metrics', color='red')
                    has_second_axis = True
                color = 'red' if 'acc' in key else 'green'
                label = key.replace('epoch_', '').replace('_', ' ').capitalize()
                ax2.plot(epochs, metrics[key], 'o-', label=label, color=color)

        if has_second_axis:
            ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()
        plt.title(f'Training Metrics Comparison - {self.train_name}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        if has_second_axis:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')

        ax1.set_xticks(self._get_epoch_range(len(epochs)))

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_metrics_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Metrics comparison plot saved to {save_path}")


    def plot_summary(self):
        """
        Create a compact dashboard-style figure with key metrics subplots.
        """

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_list = [m[1] for m in all_metrics if isinstance(m[1], dict)]
        metrics = self._convert_metrics_list_to_dict(metrics_list)

        if not metrics:
            print("No valid metrics found.")
            return

        has_loss = 'epoch_loss' in metrics
        has_time = 'epoch_time' in metrics
        has_train_acc = 'epoch_train_acc' in metrics
        has_val_acc = 'epoch_val_acc' in metrics
        has_lr = 'epoch_lr' in metrics

        plot_count = sum([has_loss, has_time, has_train_acc or has_val_acc, has_lr])
        if plot_count == 0:
            print("No plottable metrics found.")
            return

        rows = min(3, plot_count)
        cols = (plot_count + rows - 1) // rows
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        current_ax = 0

        if has_loss:
            loss = metrics['epoch_loss']
            epochs = list(range(1, len(loss) + 1))
            ax = axes[current_ax]
            ax.plot(epochs, loss, 'o-', color='blue')
            min_loss = min(loss)
            min_epoch = loss.index(min_loss) + 1
            ax.scatter([min_epoch], [min_loss], color='green', s=50, zorder=5)
            ax.set_title('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            current_ax += 1

        if has_time:
            epoch_time = metrics['epoch_time']
            epochs = list(range(1, len(epoch_time) + 1))
            ax = axes[current_ax]
            ax.bar(epochs, epoch_time, color='skyblue')
            avg_time = np.mean(epoch_time)
            ax.axhline(y=avg_time, color='r', linestyle='-', label=f'Avg: {avg_time:.2f}s')
            ax.set_title('Time per Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (s)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            ax.legend()
            current_ax += 1

        if has_train_acc or has_val_acc:
            acc_len = len(metrics.get('epoch_train_acc', metrics.get('epoch_val_acc', [])))
            epochs = list(range(1, acc_len + 1))
            ax = axes[current_ax]
            if has_train_acc:
                ax.plot(epochs, metrics['epoch_train_acc'], 'o-', color='blue', label='Train Acc')
            if has_val_acc:
                val_acc = metrics['epoch_val_acc']
                ax.plot(epochs, val_acc, 'o-', color='red', label='Val Acc')
                best_acc = max(val_acc)
                best_epoch = val_acc.index(best_acc) + 1
                ax.scatter([best_epoch], [best_acc], color='green', s=50, zorder=5)
            ax.set_title('Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            current_ax += 1

        if has_lr:
            lr = metrics['epoch_lr']
            epochs = list(range(1, len(lr) + 1))
            ax = axes[current_ax]
            ax.plot(epochs, lr, 'o-', color='purple')
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('LR')
            ax.grid(True, linestyle='--', alpha=0.7)
            if max(lr) > min(lr) * 10:
                ax.set_yscale('log')
            ax.set_xticks(self._get_epoch_range(len(epochs)))
            current_ax += 1

        for i in range(current_ax, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.train_name}_summary.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Summary plot saved to {save_path}")

    def plot_all(self):
        """Generate all available plots."""
        print(f"Generating all plots for {self.train_name}...")
        self.plot_epoch_loss()
        self.plot_epoch_time()
        self.plot_epoch_throughput()
        self.plot_accuracy()
        self.plot_learning_rate()
        self.plot_metric_comparison()
        self.plot_summary()
        print("All plots generated successfully!")

    def plot_all_per_rank(self):
        """Generate all available plots."""
        print(f"Generating all plots for {self.train_name}...")
        self.plot_epoch_loss()
        self.plot_epoch_time()
        self.plot_epoch_throughput()
        self.plot_accuracy()
        self.plot_accuracy_per_client()
        self.plot_learning_rate()
        self.plot_metric_comparison_per_rank()
        self.plot_summary_per_rank()
        print("All plots generated successfully!")

    def plot_summary_per_rank(self):
        """Generate summary plots for each rank."""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        for filename, metrics in all_metrics:
            tag = os.path.splitext(os.path.basename(filename))[0]
            print(f"Generating summary for {tag}...")

            has_loss = 'epoch_loss' in metrics
            has_time = 'epoch_time' in metrics
            has_train_acc = 'epoch_train_acc' in metrics
            has_val_acc = 'epoch_val_acc' in metrics
            has_lr = 'epoch_lr' in metrics

            if not (has_loss or has_time or has_train_acc or has_val_acc or has_lr):
                print(f"No plottable metrics found in {filename}")
                continue

            plot_count = sum([has_loss, has_time, (has_train_acc or has_val_acc), has_lr])
            rows = min(3, plot_count)
            cols = (plot_count + rows - 1) // rows
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = np.atleast_1d(axes).flatten()

            current_ax = 0
            epochs = list(range(1, len(metrics.get('epoch_loss', [])) + 1))

            if has_loss:
                ax = axes[current_ax]
                ax.plot(epochs, metrics['epoch_loss'], 'o-', color='blue')
                min_loss = min(metrics['epoch_loss'])
                min_epoch = metrics['epoch_loss'].index(min_loss) + 1
                ax.scatter([min_epoch], [min_loss], color='green', s=50)
                ax.set_title('Training Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xticks(self._get_epoch_range(len(epochs)))
                current_ax += 1

            if has_time:
                ax = axes[current_ax]
                ax.bar(epochs, metrics['epoch_time'], color='skyblue')
                avg_time = np.mean(metrics['epoch_time'])
                ax.axhline(y=avg_time, color='r', linestyle='-', label=f'Avg: {avg_time:.2f}s')
                ax.set_title('Time per Epoch')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Time (s)')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xticks(self._get_epoch_range(len(epochs)))
                ax.legend()
                current_ax += 1

            if has_train_acc or has_val_acc:
                ax = axes[current_ax]
                if has_train_acc:
                    ax.plot(epochs, metrics['epoch_train_acc'], 'o-', color='blue', label='Train')
                if has_val_acc:
                    ax.plot(epochs, metrics['epoch_val_acc'], 'o-', color='red', label='Val')
                    best_acc = max(metrics['epoch_val_acc'])
                    best_epoch = metrics['epoch_val_acc'].index(best_acc) + 1
                    ax.scatter([best_epoch], [best_acc], color='green', s=50)
                ax.set_title('Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                ax.set_xticks(self._get_epoch_range(len(epochs)))
                current_ax += 1

            if has_lr:
                ax = axes[current_ax]
                ax.plot(epochs, metrics['epoch_lr'], 'o-', color='purple')
                ax.set_title('Learning Rate')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('LR')
                ax.grid(True, linestyle='--', alpha=0.7)
                if max(metrics['epoch_lr']) > min(metrics['epoch_lr']) * 10:
                    ax.set_yscale('log')
                ax.set_xticks(self._get_epoch_range(len(epochs)))
                current_ax += 1

            for i in range(current_ax, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, f"{tag}_summary.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Summary plot saved to {save_path}")

    def plot_metric_comparison_per_rank(self):
        """Generate comparison plot for each rank."""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        for filename, metrics in all_metrics:
            if 'epoch_loss' not in metrics:
                print(f"No epoch_loss data in {filename}")
                continue

            epoch_metrics = [k for k in metrics.keys() if k.startswith('epoch_') and k != 'epoch_time']
            if len(epoch_metrics) < 2:
                print(f"Not enough metrics for comparison in {filename}")
                continue

            tag = os.path.splitext(os.path.basename(filename))[0]

            epochs = list(range(1, len(metrics['epoch_loss']) + 1))

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='blue')
            ax1.plot(epochs, metrics['epoch_loss'], 'o-', color='blue', label='Loss')
            ax1.tick_params(axis='y', labelcolor='blue')

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
                    ax2.tick_params(axis='y', labelcolor='red')

            fig.tight_layout()
            ax1.set_xticks(self._get_epoch_range(len(epochs)))

            lines1, labels1 = ax1.get_legend_handles_labels()
            if has_second_axis:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax1.legend(loc='upper right')

            plt.title(f'Metric Comparison - {tag}')
            save_path = os.path.join(self.plots_dir, f"{tag}_metrics_comparison.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Metrics comparison plot saved to {save_path}")

