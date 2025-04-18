import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_log_files(log_dir='./logs'):
    """Parse log files from all ranks and extract training metrics"""
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} not found")

    log_files = [f for f in os.listdir(log_dir) if f.startswith('training_rank_') and f.endswith('.log')]

    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    print(f"Found {len(log_files)} log files to process")

    # Data structures to store metrics
    metrics = {
        'loss': defaultdict(list),  # {epoch: [loss values]}
        'imgs_per_sec': defaultdict(list),  # {epoch: [throughput values]}
        'epoch_times': [],  # List of epoch durations
        'rank_data': defaultdict(lambda: defaultdict(list))  # {rank: {metric: [values]}}
    }

    # Regex patterns for extracting metrics
    loss_pattern = re.compile(r'Epoch: (\d+)/\d+ \| Batch: \d+/\d+ \| Loss: ([0-9.]+) \| Images/sec: ([0-9.]+)')
    epoch_time_pattern = re.compile(r'Epoch (\d+) completed in ([0-9.]+) seconds')
    rank_pattern = re.compile(r'Rank rank_(\d+)')

    # Process each log file
    for log_file in log_files:
        rank = int(log_file.split('_')[-1].split('.')[0])
        file_path = os.path.join(log_dir, log_file)

        with open(file_path, 'r') as f:
            for line in f:
                # Extract loss and throughput
                loss_match = loss_pattern.search(line)
                if loss_match:
                    epoch = int(loss_match.group(1))
                    loss = float(loss_match.group(2))
                    throughput = float(loss_match.group(3))

                    metrics['loss'][epoch].append(loss)
                    metrics['imgs_per_sec'][epoch].append(throughput)
                    metrics['rank_data'][rank]['loss'].append(loss)
                    metrics['rank_data'][rank]['throughput'].append(throughput)

                # Extract epoch completion times
                epoch_time_match = epoch_time_pattern.search(line)
                if epoch_time_match:
                    epoch = int(epoch_time_match.group(1))
                    duration = float(epoch_time_match.group(2))
                    metrics['epoch_times'].append((epoch, duration))

    return metrics


def generate_plots(metrics, output_dir='./plots'):
    """Generate training plots from parsed metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loss over time (all epochs)
    plt.figure(figsize=(10, 6))
    epochs = sorted(metrics['loss'].keys())
    avg_losses = [np.mean(metrics['loss'][epoch]) for epoch in epochs]

    plt.plot(epochs, avg_losses, 'b-', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'loss_over_epochs.png'))

    # 2. Images per second (throughput) over time
    plt.figure(figsize=(10, 6))
    throughput = [np.mean(metrics['imgs_per_sec'][epoch]) for epoch in epochs]

    plt.plot(epochs, throughput, 'g-', marker='o')
    plt.title('Training Throughput Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Images Per Second')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'throughput_over_epochs.png'))

    # 3. Epoch completion times
    if metrics['epoch_times']:
        plt.figure(figsize=(10, 6))
        epoch_nums, durations = zip(*sorted(metrics['epoch_times']))

        plt.bar(epoch_nums, durations, color='purple', alpha=0.7)
        plt.title('Time Taken per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Duration (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.savefig(os.path.join(output_dir, 'epoch_durations.png'))

    # 4. Loss comparison across different ranks
    plt.figure(figsize=(12, 8))
    for rank, data in metrics['rank_data'].items():
        if data['loss']:  # Check if there's loss data for this rank
            # Take a sample of losses to avoid overcrowding the plot
            step = max(1, len(data['loss']) // 100)  # Sample to get around 100 points
            losses = data['loss'][::step]
            steps = list(range(len(losses)))

            plt.plot(steps, losses, label=f'Rank {rank}', alpha=0.7)

    plt.title('Loss Comparison Across Ranks')
    plt.xlabel('Training Step (sampled)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'loss_by_rank.png'))

    # 5. Combined multi-metric plot
    plt.figure(figsize=(15, 10))

    # Normalize values for comparison
    norm_loss = [l / max(avg_losses) for l in avg_losses]
    norm_throughput = [t / max(throughput) for t in throughput]

    plt.subplot(2, 1, 1)
    plt.plot(epochs, norm_loss, 'b-', marker='o', label='Normalized Loss')
    plt.plot(epochs, norm_throughput, 'g-', marker='s', label='Normalized Throughput')
    plt.title('Training Metrics Over Time (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, avg_losses, 'b-', marker='o', label='Loss')
    plt.title('Training Loss (Actual Values)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))

    print(f"All plots saved to {output_dir}")


def plot_data():
    parser = argparse.ArgumentParser(description='Generate plots from distributed training logs')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory containing the training log files')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save the generated plots')
    args = parser.parse_args()

    try:
        print(f"Parsing log files from {args.log_dir}")
        metrics = parse_log_files(args.log_dir)
        generate_plots(metrics, args.output_dir)
        print("Plot generation completed successfully!")
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
