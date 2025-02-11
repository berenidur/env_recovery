import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import time

from unets import UNet
from utils import *

modelname = 'unet_v0.1'
h5_path = '../data/dataoncosalud/res_valid/comp_env_data.h5'
dataset = 'comp_env_interp_1'
n = 57 # H,W of each window
checkpoint_path = f'models/{modelname}_latest.pth'
resume_training = os.path.exists(checkpoint_path)

# Define a dummy dataset
class H5Dataset(Dataset):
    def __init__(self, lstgroups, dataset, Q1, transform=None):
        self.lstgroups = lstgroups
        self.dataset = dataset
        self.transform = transform
        self.Q1 = Q1

    def __len__(self):
        return len(self.lstgroups)

    def __getitem__(self, idx):
        x = load_h5_dataset(h5_path, self.lstgroups[idx], self.dataset)[:self.Q1,:]/255
        y = load_h5_dataset(h5_path, self.lstgroups[idx], 'validRS')[:self.Q1,:]
        y[-(n-1)//2:, :] = 0    # Set the last n//2 rows to 0 due to [:self.Q1,:] in x

        if x.shape[1] == 512:
            x = x[:, ::2]
        # if y.shape[1] == 512:
            y = y[:, ::2]

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # print(f'{idx}\tx shape: {x.shape}\ty shape: {y.shape}')

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
    
def to_tensor(image):
    image = torch.tensor(image, dtype=torch.float32)
    return image

with open('data_splits.pkl', 'rb') as f:
    data_splits = pickle.load(f)

train_files = data_splits['train_files']
val_files   = data_splits['val_files']
test_files  = data_splits['test_files']
Q1          = data_splits['Q1']

transform = to_tensor
test_files.sort()
train_dataset = H5Dataset(lstgroups=train_files, dataset=dataset, Q1=Q1, transform=transform)
val_dataset = H5Dataset(lstgroups=val_files, dataset=dataset, Q1=Q1, transform=transform)
test_dataset = H5Dataset(lstgroups=test_files, dataset=dataset, Q1=Q1, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
print(device)
start_epoch = 1

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# To store losses
history = {
    "train_loss": [],
    "val_loss": [],
    "epoch_time": []
}

if resume_training:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    print(f"Resuming training, starting from epoch {start_epoch}...")
    for i in range(start_epoch-1):
        print(f"Epoch {i+1} - Train Loss: {history['train_loss'][i]:.4f} - Validation Loss: {history['val_loss'][i]:.4f} - Time: {disp_time(history['epoch_time'][i])}")

# Training and validation loop
epochs = 20
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

    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, checkpoint_path)

    # Plot losses and test model every 10 epochs
    if epoch % 5 == 0:
        plot_losses(history, epoch, modelname)
        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, f'models/{modelname}_epoch_{epoch}.pth')
        # Save training history
        with open(f'models/{modelname}_history_epoch_{epoch}.pkl', 'wb') as f:
            pickle.dump(history, f)

        # Test the model
        test_outputs = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_outputs.append(outputs.cpu().numpy())
        test_outputs = np.concatenate(test_outputs, axis=0)
        checkpath(f'outputs/{modelname}/')
        np.save(f'outputs/{modelname}/test_outputs_epoch_{epoch}.npy', test_outputs)
        
        print(f"Test outputs saved for epoch {epoch}")
