import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 style CNN for CIFAR-100.

    - Feature extractor: three Conv(3x3, padding=1) blocks with BatchNorm + ReLU.
      MaxPool(2x2) follows conv1 and conv2; conv3 is unpooled.
    - Size handling: AdaptiveAvgPool2d → (4,4) removes hard assumptions on input
      (for 32x32 images: 32→16→8→4 spatially).
    - Classifier: Flatten (128*4*4=2048) → Linear(512) → Dropout(0.5) → Linear(num_classes).
    - Outputs raw logits (no softmax); pair with CrossEntropyLoss.
    - BatchNorm improves stability in (distributed) training and supports higher learning rates.

    Shape sketch for 32x32 input:
      32x32x3 → conv1 → 32x32x32 → pool → 16x16x32
      → conv2 → 16x16x64 → pool → 8x8x64
      → conv3 → 8x8x128 → adaptive pool → 4x4x128 → flatten → 512 → num_classes
    """
    def __init__(self, num_classes=100):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(64) 

        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        self.bn3 = nn.BatchNorm2d(128)
        
       
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
      
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 

    def forward(self, x):
      
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

      
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))

      
        x = self.adaptive_pool(x)

      
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))

      
        x = self.dropout(x)

      
        x = self.fc2(x)
        return x