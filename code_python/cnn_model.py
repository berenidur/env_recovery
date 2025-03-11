import torch
import torch.nn as nn

class cnn_v0_1(nn.Module):
    def __init__(self):
        super(cnn_v0_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjusted for 57x57 input
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Output: (32, 28, 28)
        x = self.pool(torch.relu(self.conv2(x)))  # Output: (64, 14, 14)
        x = torch.flatten(x, 1)  # Flatten to (64*14*14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x # BCEWithLogitsLoss() does not need Sigmoid activation

class cnn_v0_2(nn.Module):
    def __init__(self):
        super(cnn_v0_2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)         # increases validation loss
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  # Adjusted for input size 57x57
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (32, 28, 28)
        x = self.pool(torch.relu(self.conv2(x)))  # (64, 14, 14)
        x = self.pool(torch.relu(self.conv3(x)))  # (128, 7, 7)
        x = self.pool(torch.relu(self.conv4(x)))  # (256, 3, 3)
        
        x = torch.flatten(x, 1)  # Flatten to (256*3*3)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid for BCEWithLogitsLoss
        
        return x

class cnn_breast_v0_2(nn.Module):
    def __init__(self):
        super(cnn_breast_v0_2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)         # increases validation loss
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  # Adjusted for input size 57x57
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (32, 28, 28)
        x = self.pool(torch.relu(self.conv2(x)))  # (64, 14, 14)
        x = self.pool(torch.relu(self.conv3(x)))  # (128, 7, 7)
        x = self.pool(torch.relu(self.conv4(x)))  # (256, 3, 3)
        
        x = torch.flatten(x, 1)  # Flatten to (256*3*3)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid for BCEWithLogitsLoss
        
        return x
