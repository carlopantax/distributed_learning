import os
import json
import time
import logging
import math
from collections import defaultdict


class TrainingLogger:
    def __init__(self, log_dir='logs', train_name=None, client_id=0, num_clients=1):
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
        """Logs a message with the specified level."""
        if level.lower() == 'info':
            self.logger.info(message)
        elif level.lower() == 'warning':
            self.logger.warning(message)
        elif level.lower() == 'error':
            self.logger.error(message)
        elif level.lower() == 'debug':
            self.logger.debug(message)


    def log_metrics(self, epoch, batch_idx, loss, batch_size, train_size, extras=None):
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
            f"[Client {self.client_id}/{self.num_clients}] "
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
        self.metrics.setdefault('epoch_loss', []).append(epoch_loss)
        self.metrics.setdefault('epoch_time', []).append(epoch_time)

        if world_size is not None:
            self.metrics.setdefault('epoch_world_size', []).append(world_size)

        if extras:
            for key, value in extras.items():
                self.metrics.setdefault(f"epoch_{key}", []).append(value)

        log_msg = (
            f"[Client {self.client_id}/{self.num_clients}] "
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
        throughput = total_images / total_time

        self.metrics['total_time'] = total_time
        self.metrics['throughput'] = throughput
        self.metrics['client_id'] = self.client_id

        if num_clients is not None:
            self.metrics['num_clients'] = num_clients
        else:
            self.metrics['num_clients'] = self.num_clients
            num_clients = self.num_clients

        self.log(f"[Client {self.client_id}/{num_clients}] Training completed in {total_time:.2f} seconds.")
        self.log(f"[Client {self.client_id}/{num_clients}] Average throughput: {throughput:.2f} Images/sec")

        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.log(f"Metrics saved to {self.metrics_file}")
