import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import CNN  # Make sure this matches your training model

def evaluate():
    # --- Config ---
    model_path = 'cifar100_distributed_cpu_model.pth'
    batch_size = 128
    device = torch.device("cpu")  # or "cuda" if you're testing on GPU

    # --- Load CIFAR-100 test dataset ---
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Load model ---
    model = CNN(num_classes=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Evaluation loop ---
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error in batch {i}: {e}")

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    evaluate()
