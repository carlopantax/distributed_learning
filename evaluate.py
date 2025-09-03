import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.lenet import LeNet5

def evaluate(model_path, device='cpu'):
    """
    Evaluate a saved LeNet-5 checkpoint on the CIFAR-100 test set.

    What this script does (essentials):
    - Builds the standard CIFAR-100 test transform (same normalization as training).
    - Loads the 10k-image CIFAR-100 test split and iterates in batches.
    - Restores a LeNet5(num_classes=100) from a .pth state_dict and moves it to the chosen device.
    - Runs a no-grad forward pass in eval mode and reports final Top-1 accuracy.

    """
    print(f"Evaluating model from: {model_path}")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model = LeNet5(num_classes=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on CIFAR-100')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    evaluate(model_path=args.model, device=args.device)
