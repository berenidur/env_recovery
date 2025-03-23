import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import time

import model_autoencoders
from utils import *

# Model parameters
modelname = 'autoencoder_breast_v0.3'
checkpath(f'models/{modelname}/')
# h5_path = '../data/dataoncosalud/res_valid/comp_env_data.h5'
# dataset = 'comp_env_interp_1'
n = 57  # Height, Width of each window
checkpoint_path = f'models/{modelname}/latest.pth'
resume_training = os.path.exists(checkpoint_path)

class H5Dataset_windows(Dataset):
    def __init__(self, windows, transform=None):
        self.windows = windows
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        comp_env_window = self.windows[idx].comp_env_window
        x_R = np.mean(comp_env_window) / np.std(comp_env_window)
        x_S = np.mean((comp_env_window - np.mean(comp_env_window))**3) / (np.std(comp_env_window)**3)


        y_R = self.windows[idx].R
        y_S = self.windows[idx].S
        y_beta = self.windows[idx].beta
        y_k = self.windows[idx].k

        x = np.array([x_R, x_S], dtype=np.float32)
        y = np.array([y_R, y_S, y_beta, y_k], dtype=np.float32)

        x = np.expand_dims(x, axis=0)  # Add channel dimension
        y = np.expand_dims(y, axis=0)  # For consistency

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y


def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

# Load dataset splits
with open('breast_noNaN_data_arrays_CNN.pkl', 'rb') as f:
    data_splits = pickle.load(f)

train_files  = data_splits['train_windows']
val_files = data_splits['val_windows']

train_dataset = H5Dataset_windows(train_files, transform=to_tensor)
val_dataset = H5Dataset_windows(val_files, transform=to_tensor)

batch_size = 4096
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Initialize Model, Loss & Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ModelClass = getattr(model_autoencoders, modelname.replace(".", "_"))  # Get the class
model = ModelClass().to(device)  # Instantiate and move to device
print(f"Using device: {device}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# History tracking
history = {
    "train_loss": [],
    "val_loss": [],
    "epoch_time": []
}

# Load checkpoint if resuming training
start_epoch = 1
if resume_training:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    print(f"Resuming training from epoch {start_epoch}...")


# Training & Validation Loop
epochs = 600
for epoch in range(start_epoch, start_epoch + epochs):
    start_time = time.time()
    print(f'Epoch {epoch}/{start_epoch + epochs - 1}', end='', flush=True)

    # Training
    model.train()
    train_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    history["train_loss"].append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    history["val_loss"].append(val_loss)

    epoch_time = time.time() - start_time
    history["epoch_time"].append(epoch_time)
    print(f" - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Time: {disp_time(epoch_time)}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, checkpoint_path)

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        plot_losses(history, epoch, modelname, start_epoch=10)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, f'models/{modelname}/epoch_{epoch}.pth')