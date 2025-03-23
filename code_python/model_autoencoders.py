import torch
import torch.nn as nn
import torch.nn.functional as F

class autoencoder_breast_v0_1(nn.Module):
    def __init__(self, b=3): # b is the bottleneck size from Tehrani et al. 2024
        super(autoencoder_breast_v0_1, self).__init__()
        
        # Encoder
        # self.enc1 = nn.Linear(64, 32)
        self.enc1 = nn.Linear(2, 32)
        self.enc2 = nn.Linear(32, 32)
        self.enc3 = nn.Linear(32, b)
        
        # Decoder
        self.dec1 = nn.Linear(b, 32)
        self.dec2 = nn.Linear(32, 32)
        # self.dec3 = nn.Linear(32, 8)
        self.dec3 = nn.Linear(32, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Encoding
        x = F.leaky_relu(self.enc1(x))
        x = self.dropout(F.leaky_relu(self.enc2(x)))  # Dropout on second encoder layer
        x = F.leaky_relu(self.enc3(x))
        
        # Decoding
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x = self.dec3(x)  # No activation on last layer
        
        return x

class autoencoder_breast_v0_2(nn.Module):
    def __init__(self, b=3): # b is the bottleneck size from Tehrani et al. 2024
        super(autoencoder_breast_v0_2, self).__init__()
        
        # Encoder
        # self.enc1 = nn.Linear(64, 32)
        self.enc1 = nn.Linear(2, 32)
        self.enc2 = nn.Linear(32, 32)
        self.enc3 = nn.Linear(32, b)
        
        # Decoder
        self.dec1 = nn.Linear(b, 32)
        self.dec2 = nn.Linear(32, 32)
        # self.dec3 = nn.Linear(32, 8)
        self.dec3 = nn.Linear(32, 4)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Encoding
        x = F.leaky_relu(self.enc1(x))
        x = self.dropout(F.leaky_relu(self.enc2(x)))  # Dropout on second encoder layer
        x = F.leaky_relu(self.enc3(x))
        
        # Decoding
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x = self.dec3(x)  # No activation on last layer
        
        return x

class autoencoder_breast_v0_3(nn.Module):
    def __init__(self, b1=4, b2=4):  # Two latent spaces
        super(autoencoder_breast_v0_3, self).__init__()
        
        # Encoder
        self.enc1 = nn.Linear(2, 32)
        self.enc2 = nn.Linear(32, 32)
        
        # Two separate latent spaces
        self.enc3_1 = nn.Linear(32, b1)
        self.enc3_2 = nn.Linear(32, b2)
        
        # Decoder
        self.dec1 = nn.Linear(b1 + b2, 32)  # Merge both latent spaces
        self.dec2 = nn.Linear(32, 32)
        self.dec3 = nn.Linear(32, 4)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Encoding
        x = F.leaky_relu(self.enc1(x))
        x = self.dropout(F.leaky_relu(self.enc2(x)))  # Dropout on second encoder layer
        
        # Two separate latent spaces
        z1 = F.leaky_relu(self.enc3_1(x))
        z2 = F.leaky_relu(self.enc3_2(x))
        
        # Merge latent spaces
        z = torch.cat((z1, z2), dim=2)
        
        # Decoding
        x = F.leaky_relu(self.dec1(z))
        x = F.leaky_relu(self.dec2(x))
        x = self.dec3(x)  # No activation on last layer
        
        return x
