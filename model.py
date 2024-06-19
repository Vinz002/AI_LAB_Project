from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch

class CarBrandDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(dataframe['brand'].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['brand']
        label_idx = self.label_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    
    
class CarBrandModel(nn.Module):
    def __init__(self, num_classes):
        super(CarBrandModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # Reducing the spatial dimensions by half
            
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
            
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
            
        # Adjust the flattened size to match the output of the conv layers
        # Input size: (600, 400)
        # After conv1+pool: (32, 300, 200)
        # After conv2+pool: (64, 150, 100)
        # After conv3+pool: (128, 75, 50)
        flattened_size = 128 * 75 * 50  # Updated to 128 * 75 * 50
            
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(p=0.5)
            
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
            
        x = torch.flatten(x, 1)  # Flatten to prepare for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
            
        return x

class SlimCarBrandModel(nn.Module):
    def __init__(self, num_classes):
        super(SlimCarBrandModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        flattened_size = 64 * 75 * 50  # Adjusted for reduced complexity
        self.fc1 = nn.Linear(flattened_size, 256)  # Smaller fully connected layer
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LogoFocusedCNN(nn.Module):
    def __init__(self, num_classes):
        super(LogoFocusedCNN, self).__init__()
            
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduce size early

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduce size early

        # Depthwise Separable Convolution to reduce parameters
        self.dw_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pw_conv = nn.Conv2d(32, 64, kernel_size=1, stride=1)

        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((20, 20))  # Adjust pooling size

        # Attention layer
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Fully connected layers with reduced neurons
        self.fc1 = nn.Linear(64 * 20 * 20, 128)  # Smaller fully connected layer
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.pw_conv(self.dw_conv(x))))

        # Apply attention
        attention_map = self.attention(x)
        x = x * attention_map

        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x