import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class H5Dataset_windows_custom_v0_1(Dataset):
    def __init__(self, windows, transform=None):
        self.windows = windows
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].comp_env_window
        # x = np.array(comp_env_window, dtype=np.float32)

        y_a0 = self.windows[idx].a_0
        y_R = self.windows[idx].R
        y_S = self.windows[idx].S

        y = np.array([y_a0, y_R, y_S], dtype=np.float32)

        # x = np.expand_dims(x, axis=0)  # Add channel dimension
        # y = np.expand_dims(y, axis=0)  # For consistency

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

# Define the deeper network
class customnetwork_v0_1(nn.Module):
    def __init__(self, latent_dim=10):
        super(customnetwork_v0_1, self).__init__()
        
        # First fully connected block
        self.fc1 = nn.Linear(57 * 57, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)  # Dropout layer

        # Latent space
        self.fc_latent = nn.Linear(128, latent_dim)
        self.bn_latent = nn.BatchNorm1d(latent_dim)

        # Second fully connected block
        self.fc4 = nn.Linear(latent_dim, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)

        # Output layer
        self.fc_out = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input

        # First block
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Latent space
        latent_vector = F.leaky_relu(self.bn_latent(self.fc_latent(x)))
        latent_value = latent_vector[:, 0].unsqueeze(1)

        # Second block
        x = F.leaky_relu(self.bn4(self.fc4(latent_vector)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)

        # Final output
        output = self.fc_out(x)

        return latent_value, output  # Return separately
