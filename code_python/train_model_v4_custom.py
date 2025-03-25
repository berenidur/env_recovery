import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import time

import custom_models
from utils import *

# Model parameters
v=0.3
modelname = f'customnetwork_v{v}'
datasetname = f'H5Dataset_windows_custom_v{v}'
checkpath(f'models/{modelname}/')
n = 57  # Height, Width of each window
checkpoint_path = f'models/{modelname}/latest.pth'
resume_training = os.path.exists(checkpoint_path)


def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

# Load dataset splits
with open('breast_noNaN_data_arrays_CNN.pkl', 'rb') as f:
    data_splits = pickle.load(f)

train_files  = data_splits['train_windows']
val_files = data_splits['val_windows']

datasetClass = getattr(custom_models, datasetname.replace(".", "_"))  # Get the class
train_dataset = datasetClass(train_files, transform=to_tensor)
val_dataset = datasetClass(val_files, transform=to_tensor)

batch_size = 8192
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Initialize Model, Loss & Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ModelClass = getattr(custom_models, modelname.replace(".", "_"))  # Get the class
model = ModelClass(latent_dim=3).to(device)  # Instantiate and move to device
print(f"Using device: {device}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
epochs = 300
for epoch in range(start_epoch, start_epoch + epochs):
    start_time = time.time()
    print(f'Epoch {epoch}/{start_epoch + epochs - 1}', end='', flush=True)

    # Training
    model.train()
    train_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        predicted_a0, outputs = model(inputs)
        loss_latent = criterion(predicted_a0[:,0], targets[:, 0])
        loss_output = criterion(outputs, targets[:, 1:])
        loss = loss_latent/100 + loss_output
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
            predicted_a0, outputs = model(inputs)
            loss_latent = criterion(predicted_a0[:,0], targets[:, 0])  
            loss_output = criterion(outputs, targets[:, 1:])
            loss = loss_latent/100 + loss_output
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
    if epoch % 10 == 0:
        plot_losses(history, epoch, modelname, start_epoch=5)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, f'models/{modelname}/epoch_{epoch}.pth')