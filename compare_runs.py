import os
import re
import argparse
import torch
from models.lenet import LeNet5
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def extract_metrics(log_file):
    total_time_pattern = re.compile(r'Training completed in ([0-9.]+) seconds')

    throughput_pattern = re.compile(r'(?:Average throughput|Images/sec): ([0-9.]+)')


    with open(log_file, 'r') as f:
        log_lines = f.readlines()

    total_time = None
    throughputs = []

    for line in log_lines:
        time_match = total_time_pattern.search(line)
        if time_match:
            total_time = float(time_match.group(1))

        throughput_match = throughput_pattern.search(line)
        if throughput_match:
            throughputs.append(float(throughput_match.group(1)))

    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else None
    return total_time, avg_throughput


def compute_accuracy(model_path, device='cpu'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model = LeNet5(num_classes=100).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def compare(centralized_log, distributed_log, centralized_model, distributed_model):
    time_c, throughput_c = extract_metrics(centralized_log)
    time_d, throughput_d = extract_metrics(distributed_log)
    acc_c = compute_accuracy(centralized_model)
    acc_d = compute_accuracy(distributed_model)

    print("\n== Training Comparison ==")
    print(f"{'Mode':<15} | {'Time (s)':<10} | {'Accuracy (%)':<13}")
    print("-" * 45)
    print(f"{'Centralized':<15} | {time_c:<10.2f} | {acc_c:<13.2f}")
    print(f"{'Distributed':<15} | {time_d:<10.2f} | {acc_d:<13.2f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare centralized vs distributed training")
    parser.add_argument('--log-centralized', type=str, default='logs_centralized/training_centralized.log')
    parser.add_argument('--log-distributed', type=str, default='logs/training_rank_0.log')
    parser.add_argument('--model-centralized', type=str, default='model_centralized.pth')
    parser.add_argument('--model-distributed', type=str, default='cifar100_model_ep10.pth')
    args = parser.parse_args()

    compare(args.log_centralized, args.log_distributed, args.model_centralized, args.model_distributed)
