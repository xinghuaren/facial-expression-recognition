# cnn.py: Proposed CNN model architecture

import torch.nn as nn
import torch.nn.functional as F

# Define a custom convolutional neural network (CNN) class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layer 1: 3 input channels, 64 output channels
        # 3x3 kernel, padding of 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # Batch normalization for the outputs of conv1
        self.bn1 = nn.BatchNorm2d(64)
        
        # Convolutional layer 2: 64 input channels, 128 output channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Batch normalization for the outputs of conv2
        self.bn2 = nn.BatchNorm2d(128)
        
        # Convolutional layer 3: 128 input channels, 256 output channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Batch normalization for the outputs of conv3
        self.bn3 = nn.BatchNorm2d(256)
        
        # Convolutional layer 4: 256 input channels, 512 output channels
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # Batch normalization for the outputs of conv4
        self.bn4 = nn.BatchNorm2d(512)
        
        # Convolutional layer 5: 512 input channels, 1024 output channels
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        # Batch normalization for the outputs of conv5
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Adaptive average pooling to reduce feature maps to size (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 2048)  # First dense layer
        self.fc2 = nn.Linear(2048, 1024)  # Second dense layer
        self.fc3 = nn.Linear(1024, 512)   # Third dense layer
        
        # Dropout layers to reduce overfitting
        self.dropout1 = nn.Dropout(0.2)  # Dropout with 20% probability
        self.dropout2 = nn.Dropout(0.5)  # Dropout with 50% probability
        
        # Output layer with 7 output neurons (e.g., for 7-class classification)
        self.fc4 = nn.Linear(512, 7)

    # Forward pass through convolutional layers with ReLU activations and batch normalization
    def forward(self, x):        
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 -> BatchNorm1 -> ReLU
        x = F.max_pool2d(x, 2)  # 2x2 Max Pooling
        x = self.dropout1(x)  # Dropout with 20%
        
        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 -> BatchNorm2 -> ReLU
        x = F.max_pool2d(x, 2)  # 2x2 Max Pooling
        x = self.dropout1(x)  # Dropout with 20%
        
        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 -> BatchNorm3 -> ReLU
        x = F.max_pool2d(x, 2)  # 2x2 Max Pooling
        x = self.dropout1(x)  # Dropout with 20%
        
        x = F.relu(self.bn4(self.conv4(x)))  # Conv4 -> BatchNorm4 -> ReLU
        x = F.max_pool2d(x, 2)  # 2x2 Max Pooling
        x = self.dropout1(x)  # Dropout with 20%
        
        x = F.relu(self.bn5(self.conv5(x)))  # Conv5 -> BatchNorm5 -> ReLU
        x = F.max_pool2d(x, 2)  # 2x2 Max Pooling
        x = self.dropout1(x)  # Dropout with 20%
        
        # Adaptive average pooling to reduce the feature maps to size (1, 1)
        x = self.pool(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activations and dropout
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout2(x)   # Dropout with 50%
        
        x = F.relu(self.fc2(x))  # Fully connected layer 2
        x = self.dropout2(x)   # Dropout with 50%
        
        x = F.relu(self.fc3(x))  # Fully connected layer 3
        x = self.dropout2(x)   # Dropout with 50%
        
        # Output layer
        x = self.fc4(x)
        
        return x