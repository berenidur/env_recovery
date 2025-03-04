import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
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
