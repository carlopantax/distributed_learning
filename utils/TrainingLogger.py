import os
import json
import time
import logging
import math
from collections import defaultdict


class TrainingLogger:
    """
    Lightweight experiment logger for (distributed) training runs.

    Responsibilities:
      - Emit human-readable logs to stdout and a per-client log file.
      - Accumulate scalar metrics in memory and persist them as a JSON file
        (one file per client).
      - Resume metric accumulation if a prior metrics JSON exists for this
        (train_name, client_id) pair.

    File layout (under `log_dir`):
      - {train_name}_client{client_id}.log          
      - {train_name}_metrics_client{client_id}.json
    """
    def __init__(self, log_dir='logs', train_name=None, client_id=0, num_clients=1):
        """
        Initialize a TrainingLogger for a single logical client/worker.

        Args:
            log_dir: Root directory where logs/metrics are stored.
            train_name: Run identifier. If None, a timestamped
                name is generated.
            client_id: Logical ID of this client/worker (used in filenames
                and log prefixes).
            num_clients: Total number of clients in the experiment.
        """
        self.client_id = client_id
        self.num_clients = num_clients
        self.metrics = defaultdict(list)
        self.start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        if train_name is None:
            train_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.exp_name = train_name
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"{train_name}_client{client_id}.log")
        self.metrics_file = os.path.join(log_dir, f"{train_name}_metrics_client{client_id}.json")

        self.logger = logging.getLogger(f"client_{client_id}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Started training for Client {client_id}")
            self.logger.info(f"Log file: {self.log_file}")

        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    previous_metrics = json.load(f)
                for key, val in previous_metrics.items():
                    if isinstance(val, list):
                        self.metrics[key].extend(val)
                    else:
                        self.metrics[key].append(val)
                self.logger.info(f"Resumed metrics from {self.metrics_file}")
            except Exception as e:
                self.logger.warning(f"Could not load previous metrics: {e}")


    def log(self, message, level='info'):
        """
        Emit a one-line message at the requested severity level.

        Args:
            message: Message to log.
            level: One of {'info','warning','error','debug'}.
        """
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
        Record per-batch metrics and persist the rolling JSON snapshot.

        Args:
            epoch: Zero-based epoch index (local/global depending on caller).
            batch_idx: Zero-based batch index within the current epoch.
            loss: Current (averaged) loss value to log.
            batch_size: Effective batch size used for this step.
            train_size: Number of samples in the client's training split.
            extras: Optional additional scalars to record. Keys are
                stored as-is;
                Example: {'train_acc': 83.2, 'lr': 0.01}
        """
        elapsed = time.time() - self.start_time
        imgs_per_sec = batch_idx * batch_size / elapsed if elapsed > 0 else 0

        local_train_size = train_size 

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
            f"[Client {self.client_id}] "
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

        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            self.log(f"[ERROR] Could not save metrics to {self.metrics_file}: {e}", level='error')


    def log_epoch(self, epoch, epoch_loss, epoch_time, world_size=None, extras=None):
        """
        Record per-epoch aggregates and persist the rolling JSON snapshot.

        Args:
            epoch: Zero-based epoch index .
            epoch_loss: Average loss over the epoch.
            epoch_time: Wall-clock seconds spent in this epoch.
            world_size: Number of clients/workers.
            extras: Optional extra scalars to store at the epoch
                granularity.
                Example: {'train_acc': 84.1, 'val_acc': 57.3, 'lr': 0.01}
        """
        self.metrics.setdefault('epoch_loss', []).append(epoch_loss)
        self.metrics.setdefault('epoch_time', []).append(epoch_time)

        if world_size is not None:
            self.metrics.setdefault('epoch_world_size', []).append(world_size)

        if extras:
            for key, value in extras.items():
                self.metrics.setdefault(f"epoch_{key}", []).append(value)

        log_msg = (
            f"[Client {self.client_id}] "
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s | "
            f"Average Loss: {epoch_loss:.4f}"
        )

        if world_size is not None:
            log_msg += f" | World Size: {world_size}"

        if extras:
            extras_str = " | ".join([f"{k}: {v:.4f}" for k, v in extras.items()])
            log_msg += f" | {extras_str}"

        self.log(log_msg)

        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            self.log(f"[ERROR] Could not save metrics to {self.metrics_file}: {e}", level='error')



    def log_training_complete(self, total_time, total_images, num_clients=None):
        """
        Finalize the run: compute and store global throughput, persist metrics.

        Args:
            total_time (float): Total wall-clock seconds for the run.
            total_images (int): Total number of training images processed by this
                client across the entire run.
            num_clients (int | None): If provided, overrides the stored
                `self.num_clients` before writing. Useful when the logger did not
                know the final world size at construction time.
        """
        throughput = total_images / total_time

        self.metrics['total_time'] = total_time
        self.metrics['throughput'] = throughput
        self.metrics['client_id'] = self.client_id

        if num_clients is not None:
            self.metrics['num_clients'] = num_clients
        else:
            self.metrics['num_clients'] = self.num_clients
            num_clients = self.num_clients

        self.log(f"[Client {self.client_id}] Training completed in {total_time:.2f} seconds.")
        self.log(f"[Client {self.client_id}] Average throughput: {throughput:.2f} Images/sec")

        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.log(f"Metrics saved to {self.metrics_file}")
