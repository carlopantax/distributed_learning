import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet5, self).__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # Input 32x32x3 → Output 32x32x32

        # Batch Normalization: aggiunta dopo ogni strato convoluzionale per
        # strabilizzare training distribuito, permettere learning di rate piu elevati e 
        # ridurre la covariante shift
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Output 16x16x64 dopo pooling
        self.bn2 = nn.BatchNorm2d(64) 

        # Strato convoluzionale extra
        # aumenta capacità del modello da 2 a 3 strati
        # migliora estrazione di feature 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # → Output 8x8x128
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Calcolato con adaptive pool per gestire le dimensioni
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Utility layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Gestione dimensioni

    def forward(self, x):
      # conv1 + bn1+ pool
      # conv1: INPUT 3 canali (RGB) -> OUTPUT 32 feature maps (dim: 32x32x32)
      # bn1: normalizza attivazioni lungo i 32 canali
      # pool: MaxPool2d 
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

      #  conv2 output 16x16x64
      # dopo  pool 8x8x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

      # conv3 8x8x128 senza pool
        x = F.relu(self.bn3(self.conv3(x)))

      # riduce qualsiasi dimensione spaziale a 4x4
      # input: 8x8x128 -> output: 4x4x128
        x = self.adaptive_pool(x)

      # trasforma 4x4x128 in un vett di 2048 elementi
      # proietta in uno spazio a 512 dim con ReLU
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))

      # durante il training spegne casualmente il 50% dei neuroni
      # previene overfitting
        x = self.dropout(x)

      # proiezione finaloe sulle 100 classi di cifar100 (nessuna softmax)
        x = self.fc2(x)
        return x