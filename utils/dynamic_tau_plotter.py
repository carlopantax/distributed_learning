import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob
import re
from utils.TrainingPlotter import TrainingPlotter


class DynamicTauPlotter(TrainingPlotter):
    """
    Extension of TrainingPlotter that adds dynamic tau specific visualizations
    """

    def plot_dynamic_tau_timeline(self):
        """Plot the evolution of tau values over time for each client"""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        plt.figure(figsize=(15, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, self.world_size))

        has_tau_data = False
        for client_id, metrics in metrics_by_rank.items():
            # Extract tau values from epoch metrics
            current_tau_values = metrics.get('epoch_current_tau', [])
            sync_rounds = metrics.get('epoch_sync_round', [])

            if not current_tau_values:
                continue

            has_tau_data = True
            epochs = list(range(1, len(current_tau_values) + 1))

            plt.step(epochs, current_tau_values, where='post', label=f'Client {client_id}',
                     color=colors[client_id % len(colors)], linewidth=2, marker='o', markersize=4)

            # Mark sync events with different markers
            if sync_rounds:
                sync_epochs = []
                sync_tau_values = []
                for i, sync_round in enumerate(sync_rounds):
                    if i < len(current_tau_values) and sync_round > 0:  # Only mark actual syncs
                        sync_epochs.append(i + 1)
                        sync_tau_values.append(current_tau_values[i])

                if sync_epochs:
                    plt.scatter(sync_epochs, sync_tau_values, color=colors[client_id % len(colors)],
                                s=50, marker='s', alpha=0.7, edgecolors='black')

        if not has_tau_data:
            print("No tau timeline data found in metrics.")
            return

        plt.title(f'Dynamic Tau Evolution - {self.train_name}')
        plt.xlabel('Training Epoch')
        plt.ylabel('Tau Value (Local Epochs per Sync)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, f"{self.train_name}_dynamic_tau_timeline.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Dynamic tau timeline plot saved to {save_path}")

    def plot_sync_frequency_analysis(self):
        """Analyze synchronization frequency patterns"""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Sync events over time
        has_sync_data = False
        for client_id, metrics in metrics_by_rank.items():
            sync_rounds = metrics.get('epoch_sync_round', [])
            epoch_times = metrics.get('epoch_time', [])

            if not sync_rounds or not epoch_times:
                continue

            has_sync_data = True
            # Find actual sync events (when sync_round changes)
            sync_epochs = []
            sync_times = []
            cumulative_time = 0

            prev_sync = 0
            for i, (sync_round, epoch_time) in enumerate(zip(sync_rounds, epoch_times)):
                cumulative_time += epoch_time
                if sync_round > prev_sync:  # New sync event
                    sync_epochs.append(i + 1)
                    sync_times.append(cumulative_time)
                    prev_sync = sync_round

            if sync_epochs:
                ax1.scatter(sync_times, [client_id] * len(sync_times),
                            label=f'Client {client_id}', s=50, alpha=0.7)

        if has_sync_data:
            ax1.set_title('Synchronization Events Over Time')
            ax1.set_xlabel('Cumulative Training Time (seconds)')
            ax1.set_ylabel('Client ID')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_yticks(range(self.world_size))

        # Plot 2: Distribution of tau values across all clients
        all_tau_values = []
        client_labels = []

        for client_id, metrics in metrics_by_rank.items():
            current_tau_values = metrics.get('epoch_current_tau', [])

            if current_tau_values:
                all_tau_values.extend(current_tau_values)
                client_labels.extend([f'Client {client_id}'] * len(current_tau_values))

        if all_tau_values:
            # Create histogram
            unique_taus = sorted(list(set(all_tau_values)))
            counts = [all_tau_values.count(tau) for tau in unique_taus]

            ax2.bar(unique_taus, counts, alpha=0.7, edgecolor='black')
            ax2.set_title('Distribution of Tau Values Across Training')
            ax2.set_xlabel('Tau Value (Local Epochs)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(unique_taus)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.train_name}_sync_analysis.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Sync frequency analysis plot saved to {save_path}")

    def plot_performance_vs_tau(self):
        """Plot performance metrics vs tau values to analyze effectiveness"""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        all_data = {'tau': [], 'val_acc': [], 'loss': [], 'grad_norm': [], 'client_id': []}

        for client_id, metrics in metrics_by_rank.items():
            current_tau = metrics.get('epoch_current_tau', [])
            val_acc = metrics.get('epoch_val_acc', [])
            loss = metrics.get('epoch_loss', [])
            grad_norm = metrics.get('epoch_grad_norm', [])

            # Align all metrics to same length
            min_len = min(len(arr) for arr in [current_tau, val_acc, loss, grad_norm] if arr)
            if min_len == 0:
                continue

            for i in range(min_len):
                all_data['tau'].append(current_tau[i] if i < len(current_tau) else current_tau[-1])
                all_data['val_acc'].append(val_acc[i] if i < len(val_acc) else None)
                all_data['loss'].append(loss[i] if i < len(loss) else None)
                all_data['grad_norm'].append(grad_norm[i] if i < len(grad_norm) else None)
                all_data['client_id'].append(client_id)

        if not all_data['tau']:
            print("No performance vs tau data found.")
            return

        # Plot relationships
        colors = plt.cm.tab10(np.linspace(0, 1, self.world_size))

        for client_id in range(self.world_size):
            client_mask = [cid == client_id for cid in all_data['client_id']]
            client_tau = [tau for tau, mask in zip(all_data['tau'], client_mask) if mask]

            if not client_tau:
                continue

            client_val_acc = [acc for acc, mask in zip(all_data['val_acc'], client_mask) if mask and acc is not None]
            client_loss = [l for l, mask in zip(all_data['loss'], client_mask) if mask and l is not None]
            client_grad_norm = [gn for gn, mask in zip(all_data['grad_norm'], client_mask) if mask and gn is not None]

            if client_val_acc and len(client_val_acc) == len(client_tau):
                ax1.scatter(client_tau, client_val_acc, label=f'Client {client_id}',
                            color=colors[client_id % len(colors)], alpha=0.6, s=30)

            if client_loss and len(client_loss) == len(client_tau):
                ax2.scatter(client_tau, client_loss, label=f'Client {client_id}',
                            color=colors[client_id % len(colors)], alpha=0.6, s=30)

            if client_grad_norm and len(client_grad_norm) == len(client_tau):
                ax3.scatter(client_tau, client_grad_norm, label=f'Client {client_id}',
                            color=colors[client_id % len(colors)], alpha=0.6, s=30)

        ax1.set_title('Validation Accuracy vs Tau')
        ax1.set_xlabel('Tau (Local Epochs)')
        ax1.set_ylabel('Validation Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title('Loss vs Tau')
        ax2.set_xlabel('Tau (Local Epochs)')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        ax3.set_title('Gradient Norm vs Tau')
        ax3.set_xlabel('Tau (Local Epochs)')
        ax3.set_ylabel('Gradient Norm')
        ax3.grid(True, alpha=0.3)

        # Summary statistics in fourth subplot
        ax4.axis('off')
        summary_text = "Dynamic Tau Summary:\n\n"

        # Calculate statistics
        unique_clients = list(set(all_data['client_id']))
        for client_id in sorted(unique_clients):
            client_mask = [cid == client_id for cid in all_data['client_id']]
            client_taus = [tau for tau, mask in zip(all_data['tau'], client_mask) if mask]

            if client_taus:
                min_tau = min(client_taus)
                max_tau = max(client_taus)
                avg_tau = sum(client_taus) / len(client_taus)

                summary_text += f"Client {client_id}:\n"
                summary_text += f"  Min tau: {min_tau}\n"
                summary_text += f"  Max tau: {max_tau}\n"
                summary_text += f"  Avg tau: {avg_tau:.1f}\n"
                summary_text += f"  Total epochs: {len(client_taus)}\n\n"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.train_name}_performance_vs_tau.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Performance vs tau plot saved to {save_path}")

    def plot_tau_adaptation_heatmap(self):
        """Create a heatmap showing tau values over training epochs for all clients"""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        # Create matrix: clients x epochs
        max_epochs = 0
        tau_data = {}

        for client_id, metrics in metrics_by_rank.items():
            current_tau = metrics.get('epoch_current_tau', [])
            if current_tau:
                tau_data[client_id] = current_tau
                max_epochs = max(max_epochs, len(current_tau))

        if not tau_data or max_epochs == 0:
            print("No tau adaptation data found.")
            return

        # Build matrix
        tau_matrix = np.zeros((len(tau_data), max_epochs))
        client_ids = sorted(tau_data.keys())

        for i, client_id in enumerate(client_ids):
            client_taus = tau_data[client_id]
            for j, tau in enumerate(client_taus):
                tau_matrix[i, j] = tau
            # Fill remaining epochs with last tau value
            if len(client_taus) < max_epochs:
                tau_matrix[i, len(client_taus):] = client_taus[-1]

        plt.figure(figsize=(15, 8))
        im = plt.imshow(tau_matrix, cmap='viridis', aspect='auto', interpolation='nearest')

        plt.colorbar(im, label='Tau Value')
        plt.title(f'Tau Adaptation Heatmap - {self.train_name}')
        plt.xlabel('Training Epoch')
        plt.ylabel('Client ID')
        plt.yticks(range(len(client_ids)), [f'Client {cid}' for cid in client_ids])

        # Add epoch ticks
        epoch_ticks = self._get_epoch_range(max_epochs)
        plt.xticks(np.array(epoch_ticks) - 1, epoch_ticks)  # -1 because imshow is 0-indexed

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.train_name}_tau_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Tau adaptation heatmap saved to {save_path}")

    def plot_convergence_comparison(self):
        """Compare convergence patterns across clients with different tau strategies"""
        all_metrics = self._load_all_metrics()
        if not all_metrics:
            print("No metrics found.")
            return

        metrics_by_rank = self._group_metrics_by_rank(all_metrics)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, self.world_size))

        for client_id, metrics in metrics_by_rank.items():
            val_acc = metrics.get('epoch_val_acc', [])
            loss = metrics.get('epoch_loss', [])

            if val_acc:
                epochs = list(range(1, len(val_acc) + 1))
                ax1.plot(epochs, val_acc, color=colors[client_id % len(colors)],
                         label=f'Client {client_id}', linewidth=2, alpha=0.8)

                # Mark best accuracy
                best_acc = max(val_acc)
                best_epoch = val_acc.index(best_acc) + 1
                ax1.scatter([best_epoch], [best_acc], color=colors[client_id % len(colors)],
                            s=100, marker='*', edgecolors='black', linewidths=1)

            if loss:
                epochs = list(range(1, len(loss) + 1))
                ax2.plot(epochs, loss, color=colors[client_id % len(colors)],
                         label=f'Client {client_id}', linewidth=2, alpha=0.8)

        ax1.set_title('Validation Accuracy Convergence')
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Validation Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title('Loss Convergence')
        ax2.set_xlabel('Training Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{self.train_name}_convergence_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Convergence comparison plot saved to {save_path}")

    def plot_all_dynamic_tau(self):
        """Generate all plots including dynamic tau specific visualizations"""
        print(f"Generating all plots including dynamic tau analysis for {self.train_name}...")

        # Original plots from parent class
        self.plot_epoch_loss()
        self.plot_accuracy_per_client()
        self.plot_learning_rate()

        # Dynamic tau specific plots
        self.plot_dynamic_tau_timeline()
        self.plot_sync_frequency_analysis()
        self.plot_performance_vs_tau()
        self.plot_tau_adaptation_heatmap()
        self.plot_convergence_comparison()

        print("All dynamic tau plots generated successfully!")