import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Double Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Attention U-Net
class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = ConvBlock(128, 64)
        
        # Output Layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat((e4, d4), dim=1))
        
        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat((e3, d3), dim=1))
        
        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat((e2, d2), dim=1))
        
        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat((e1, d1), dim=1))
        
        # Output
        out = self.out_conv(d1)
        return out
    
class AttUNet_seg_std(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttUNet_seg_std, self).__init__()

        self.attUNet = AttentionUNet(in_channels,out_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def forward(self, x):
        output_attunet = self.attUNet(x)
        binary_segmentation_map = (F.sigmoid(output_attunet) > 0.5).bool()

        masked_output = x * binary_segmentation_map
        
        batch_stds = []
        for batch_idx in range(masked_output.size(0)):  # Loop over batches
            non_zero_elements = masked_output[batch_idx][masked_output[batch_idx] != 0]  # Extract non-zero elements for the batch
            if len(non_zero_elements) > 0:
                std = torch.std(non_zero_elements)  # Standard deviation for this batch
            else:
                std = torch.tensor(0.0)  # Handle cases with no non-zero elements
            batch_stds.append(std)
        batch_stds = torch.stack(batch_stds)

        return batch_stds

# Example usage
if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256),requires_grad=True)  # Example input tensor

    model_attunet = AttentionUNet(in_channels=1, out_channels=1)  # For grayscale input, binary segmentation
    output_attunet = model_attunet(x)
    binary_segmentation_map = (F.sigmoid(output_attunet) > 0.5).bool()

    masked_output = x * binary_segmentation_map
    
    batch_stds = []
    for batch_idx in range(masked_output.size(0)):  # Loop over batches
        non_zero_elements = masked_output[batch_idx][masked_output[batch_idx] != 0]  # Extract non-zero elements for the batch
        if len(non_zero_elements) > 0:
            std = torch.std(non_zero_elements)  # Standard deviation for this batch
        else:
            std = torch.tensor(0.0)  # Handle cases with no non-zero elements
        batch_stds.append(std)

    batch_stds = torch.stack(batch_stds)
    print(batch_stds.shape)  # Expected output shape: (1, 1, 256, 256)
    print(batch_stds)
