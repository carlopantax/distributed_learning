import os
import json
import time
import logging
import math
from collections import defaultdict


class TrainingLogger:
    """
    Class to manage logging during training in a distributed setting.
    Handles log rotation, formatting, and saving metrics to file.
    """

    def __init__(self, log_dir='logs', train_name=None, rank=0, is_main_process=False, world_size=1):
        """
        Initializes the logger.

        Args:
            log_dir (str): Directory to save logs
            train_name (str): Experiment name for the log files
            rank (int): Process rank in distributed training
            is_main_process (bool): Whether this process is the main process
        """
        self.rank = rank
        self.is_main_process = is_main_process
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.world_size = world_size

        os.makedirs(log_dir, exist_ok=True)

        if train_name is None:
            train_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.exp_name = train_name
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"{train_name}_rank{rank}.log")

        self.metrics_file = os.path.join(log_dir, f"{train_name}_metrics_rank{rank}.json")

        self.logger = logging.getLogger(f"trainer_{rank}")
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers (important when resuming)
        if not self.logger.handlers:

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Started training experiment: {train_name}")
            self.logger.info(f"Log file: {self.log_file}")

            # Load metrics if resuming
            if os.path.exists(self.metrics_file):
                try:
                    with open(self.metrics_file, 'r') as f:
                        previous_metrics = json.load(f)
                    for key, val in previous_metrics.items():
                        self.metrics[key] = val if isinstance(val, list) else [val]
                    self.logger.info(f"Resumed metrics from {self.metrics_file}")
                except Exception as e:
                    self.logger.warning(f"Could not load previous metrics: {e}")

    def log(self, message, level='info'):
        """Logs a message with the specified level."""
        if level.lower() == 'info':
                self.logger.info(message)
        if self.is_main_process:
            if level.lower() == 'warning':
                self.logger.warning(message)
            elif level.lower() == 'error':
                self.logger.error(message)
            elif level.lower() == 'debug':
                self.logger.debug(message)

    def log_metrics(self, epoch, batch_idx, loss, batch_size, train_size, extras=None):
        """
        Logs training metrics for the current batch.

        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            loss (float): Current loss value
            batch_size (int): Size of the batch
            train_size (int): Total size of the training set
            extras (dict): Extra metrics to log
        """

        elapsed = time.time() - self.start_time
        
        imgs_per_sec = batch_idx * batch_size / elapsed if elapsed > 0 else 0

        if hasattr(self, 'local_train_size'):
            local_train_size = self.local_train_size
        else:
            local_train_size = train_size // self.world_size

        progress = 100. * (batch_idx + 1) * batch_size / local_train_size


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
            f"Batch: {batch_idx + 1}/{math.ceil(local_train_size / batch_size)} | "
            f"Loss: {loss:.4f} | "
            f"Images/sec: {imgs_per_sec:.2f} | "
            f"Progress: {progress:.1f}% | "
            f"Time: {elapsed:.2f}s"
        )

        if extras:
            extras_str = " | ".join([f"{k}: {v:.4f}" for k, v in extras.items()])
            log_msg += f" | {extras_str}"

        self.log(log_msg)

    def log_epoch(self, epoch, epoch_loss, epoch_time, world_size=1, extras=None):
        """
        Logs metrics for completed epoch.

        Args:
            epoch (int): Current epoch
            epoch_loss (float): Average loss for the epoch
            epoch_time (float): Time taken for the epoch
            world_size (int): Number of processes in distributed training
            extras (dict): Extra metrics to log
        """

        self.metrics['epoch_loss'].append(epoch_loss)
        self.metrics['epoch_time'].append(epoch_time)

        if extras:
            for key, value in extras.items():
                self.metrics[f"epoch_{key}"].append(value)

        log_msg = (
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s | "
            f"Average Loss: {epoch_loss:.4f}"
        )

        if extras:
            extras_str = " | ".join([f"{k}: {v:.4f}" for k, v in extras.items()])
            log_msg += f" | {extras_str}"

        self.log(log_msg)

    def log_training_complete(self, total_time, total_images, world_size=1):
        """
        Logs final training statistics.

        Args:
            total_time (float): Total training time
            total_images (int): Total number of images processed
            world_size (int): Number of processes in distributed training
        """

        throughput = total_images / total_time

        self.metrics['total_time'] = total_time
        self.metrics['throughput'] = throughput
        self.metrics['world_size'] = world_size

        self.log(f"Training completed in {total_time:.2f} seconds.")
        self.log(f"Average throughput: {throughput:.2f} Images/sec")
        self.log(f"Total processes: {world_size}")

        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.log(f"Metrics saved to {self.metrics_file}")