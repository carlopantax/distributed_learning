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

    def __init__(self, log_dir='logs', train_name=None, is_main_process=False, metrics_files=None):
        """
        Initialize the plotter.

        Args:
            log_dir (str): Directory where logs are saved
            train_name (str): Training experiment name
            is_main_process (bool): Whether this process is the main process
        """
        self.is_main_process = is_main_process

        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        self.plots_dir = os.path.join(log_dir, 'plots')

        # Set experiment name
        if train_name is None:
            metrics_files = [f for f in os.listdir(log_dir) if f.endswith('_metrics.json') or f.endswith('_metrics_combined.json')]
            metrics_files.sort()  # ensure consistent order
            if metrics_files:
                train_name = metrics_files[-1].replace('_metrics.json', '').replace('_metrics_combined.json', '')
            else:
                train_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.train_name = train_name

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

    def _load_metrics(self):
        all_metrics = []
        for path in self.metrics_files:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    all_metrics.append(data)
            else:
                print(f"[WARNING] Metrics file not found: {path}")
            if len(all_metrics) == 1:
                return all_metrics[0]  # dict centralizzato
        return all_metrics  # lista di dicts (distribuito)
        
    
        
    def _load_all_metrics(self):
        """Load and return metrics from all files in self.metrics_files."""
        all_metrics = []
        for fpath in self.metrics_files:
            if not os.path.exists(fpath):
                print(f"[WARNING] Metrics file not found: {fpath}")
                continue
            with open(fpath, 'r') as f:
                metrics = json.load(f)
                all_metrics.append((os.path.basename(fpath), metrics))
        return all_metrics

    def _convert_metrics_list_to_dict(self, metrics_list):
        """Convert list of per-epoch metrics to a dict of lists."""
        if not isinstance(metrics_list, list) or not metrics_list:
            return {}

        metrics_dict = {}
        for entry in metrics_list:
            for key, value in entry.items():
                metrics_dict.setdefault(key, []).append(value)
        return metrics_dict

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
        if not self.is_main_process:
            return

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        for filename, metrics in all_metrics:
            if 'epoch_loss' not in metrics:
                print(f"No epoch_loss data in {filename}")
                continue

            epoch_loss = metrics['epoch_loss']
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
                    smoothed = np.convolve(epoch_loss,
                                        np.ones(window_size) / window_size,
                                        mode='valid')
                    plt.plot(range(window_size, len(epochs) + 1), smoothed,
                            color='red', linestyle='--', linewidth=1.5,
                            label=f'{window_size}-Epoch Moving Avg')

            plt.title(f'Loss per Epoch - {filename}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.xticks(self._get_epoch_range(len(epochs)))

           
            tag = filename.replace('_metrics.json', '').replace('.json', '')
            save_path = os.path.join(self.plots_dir, f"{tag}_epoch_loss.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Epoch loss plot saved to {save_path}")

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
        """Plot the time taken for each epoch (aggregated from multiple ranks)."""
        if not self.is_main_process:
            return

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        # Gestione centralizzato vs distribuito
        if isinstance(all_metrics, dict):
            epoch_time = all_metrics.get('epoch_time', [])
            if not epoch_time:
                print("No epoch time data found.")
                return
            num_epochs = len(epoch_time)
            time_avg = epoch_time
        else:
            num_epochs = len(all_metrics[0][1].get('epoch_time', []))
            epoch_times = np.array([m[1].get('epoch_time', [0]*num_epochs) for m in all_metrics])
            time_avg = np.mean(epoch_times, axis=0)

        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(12, 6))
        plt.bar(epochs, time_avg, color='skyblue')
        avg_time = np.mean(time_avg)
        plt.axhline(y=avg_time, color='r', linestyle='-', label=f'Average: {avg_time:.2f}s')

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
        """Plot the throughput for each epoch (aggregated from epoch time)."""
        if not self.is_main_process:
            return

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        # Gestione centralizzato vs distribuito
        if isinstance(all_metrics, dict):
            epoch_time = all_metrics.get('epoch_time', [])
            if not epoch_time:
                print("No epoch time data found.")
                return
            num_epochs = len(epoch_time)
            avg_times = epoch_time
        else:
            num_epochs = len(all_metrics[0][1].get('epoch_time', []))
            epoch_times = np.array([m[1].get('epoch_time', [0]*num_epochs) for m in all_metrics])
            avg_times = np.mean(epoch_times, axis=0)

        throughputs = [1.0 / t if t > 0 else 0 for t in avg_times]

        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(12, 6))
        plt.bar(epochs, throughputs, color='lightgreen')
        avg_throughput = np.mean(throughputs)
        plt.axhline(y=avg_throughput, color='r', linestyle='-', label=f'Average: {avg_throughput:.4f}')

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



    def plot_accuracy(self):
        """Plot the training and validation accuracy over epochs (aggregated or centralized)."""
        if not self.is_main_process:
            return

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        if isinstance(all_metrics, dict):  # Centralized case
            train_avg = all_metrics.get('epoch_train_acc', [])
            val_avg = all_metrics.get('epoch_val_acc', [])
            if len(train_avg) == 0 and len(val_avg) == 0:
                print("No accuracy data found.")
                return
            num_epochs = len(train_avg)
        else:  # Distributed case
            num_epochs = len(all_metrics[0][1].get('epoch_train_acc', []))
            train_acc_all = np.array([m[1].get('epoch_train_acc', [0] * num_epochs) for m in all_metrics])
            val_acc_all = np.array([m[1].get('epoch_val_acc', [0] * num_epochs) for m in all_metrics])
            train_avg = np.mean(train_acc_all, axis=0)
            val_avg = np.mean(val_acc_all, axis=0)

        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(12, 6))

        if train_avg is not None and len(train_avg) > 0:
            plt.plot(epochs, train_avg, 'o-', color='blue', label='Train Accuracy (avg)')
        if val_avg is not None and len(val_avg) > 0:
            plt.plot(epochs, val_avg, 'o-', color='red', label='Validation Accuracy (avg)')
            best_acc = max(val_avg)
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
        """Plot the learning rate over epochs (centralized or from rank 0 in distributed setup)."""
        if not self.is_main_process:
            return

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        # Estrazione corretta del dizionario
        if isinstance(all_metrics, dict):  # Centralized
            epoch_lr = all_metrics.get('epoch_lr', None)
        else:  # Distributed (prendiamo il rank 0 come riferimento per lr)
            epoch_lr = all_metrics[0][1].get('epoch_lr', None)

        if epoch_lr is None or len(epoch_lr) == 0:
            print("No learning rate data found.")
            return

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
        """Plot multiple metrics on the same graph for comparison."""
        if not self.is_main_process:
            return

        metrics_raw = self._load_metrics()
        if isinstance(metrics_raw, list):
            metrics_raw = self._convert_metrics_list_to_dict(metrics_raw)
        else:
            metrics_raw = metrics_raw  # già nel formato corretto
        metrics = metrics_raw


        if not metrics:
            print("No metrics found.")
            return

        epoch_metrics = [key for key in metrics.keys() if key.startswith('epoch_') and key != 'epoch_time']

        if not epoch_metrics:
            print("No epoch metrics found for comparison.")
            return


        keys = ['loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
        available = [k for k in keys if k in metrics and isinstance(metrics[k], list)]
        if len(available) < 2:
            print("Not enough metrics for comparison.")
            return

        epochs = list(range(1, len(metrics['loss']) + 1))
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')

        if 'loss' in metrics:
            ax1.plot(epochs, metrics['loss'], 'o-', color='blue', label='Train Loss')

        if 'val_loss' in metrics:
            ax1.plot(epochs, metrics['val_loss'], 's--', color='cyan', label='Val Loss')

        ax1.tick_params(axis='y', labelcolor='blue')

        has_second_axis = False
        for key in available:
            if key != 'loss':
                if not has_second_axis:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Other Metrics', color='red')
                    has_second_axis = True
                color = 'red' if key == 'val_acc' else 'green'
                label = key.replace('_', ' ').capitalize()
                ax2.plot(epochs, metrics[key], 'o-', color=color, label=label)

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
        """Generate a summary plot with multiple subplots for key metrics."""
        if not self.is_main_process:
            return

        metrics_raw = self._load_metrics()
        if isinstance(metrics_raw, list):
            metrics_raw = self._convert_metrics_list_to_dict(metrics_raw)
        metrics = metrics_raw

        if not metrics:
            print("No metrics found.")
            return

        # Verifica disponibilità metriche
        has_loss = 'loss' in metrics
        has_time = 'epoch_time' in metrics
        has_train_acc = 'train_acc' in metrics
        has_val_acc = 'val_acc' in metrics
        has_lr = 'lr' in metrics

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
            loss = metrics['loss']
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
            acc_len = len(metrics.get('train_acc', metrics.get('val_acc', [])))
            epochs = list(range(1, acc_len + 1))
            ax = axes[current_ax]
            if has_train_acc:
                ax.plot(epochs, metrics['train_acc'], 'o-', color='blue', label='Train Acc')
            if has_val_acc:
                val_acc = metrics['val_acc']
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
            lr = metrics['lr']
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

        # Disattiva assi vuoti
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

    def plot_all_per_rank(self):
        """Generate all available plots."""
        if not self.is_main_process:
            return

        print(f"Generating all plots for {self.train_name}...")
        self.plot_epoch_loss()
        self.plot_epoch_time()
        self.plot_epoch_throughput()
        self.plot_accuracy()
        self.plot_learning_rate()
        self.plot_metric_comparison_per_rank()
        self.plot_summary_per_rank()
        print("All plots generated successfully!")

    def plot_summary_per_rank(self):
        """Generate summary plots for each rank."""
        if not self.is_main_process:
            return

        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        for filename, metrics in all_metrics:
            tag = filename.replace('_metrics.json', '').replace('.json', '')
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
        if not self.is_main_process:
            return

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

            tag = filename.replace('_metrics.json', '').replace('.json', '')
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

            # Combined legend
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

