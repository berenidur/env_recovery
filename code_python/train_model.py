import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from unets import AttUNet_seg_std
import scipy.io
import pickle
import h5py

std_const=torch.pi/2/np.sqrt(6)

modelname='unet'
def idx2datapathkey(simu):
    h5dir='/data/data_'
    if 1<=simu and simu<=1000:
        datapath=h5dir+'00001-01000.h5'
    elif 1001<=simu and simu<=2000:
        datapath=h5dir+'01001-02000.h5'
    elif 2001<=simu and simu<=3000:
        datapath=h5dir+'02001-03000.h5'
    elif 3001<=simu and simu<=4000:
        datapath=h5dir+'03001-04000.h5'

    datakey = 'data' + str(simu).zfill(5)

    return datapath, datakey

# Define a dummy dataset
class H5Dataset(Dataset):
    def __init__(self, lstfiles, transform=None):
        self.lstfiles = lstfiles
        self.transform = transform

    def __len__(self):
        return len(self.lstfiles)

    def __getitem__(self, idx):
        datapath, datakey = idx2datapathkey(self.lstfiles[idx])
        with h5py.File(datapath, 'r') as f:
            x = np.array(f[datakey]['input'])
            y = np.array(f[datakey]['target'])

        x/=256
        y/=256

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

transform = to_tensor
test_files.sort()
train_dataset = H5Dataset(lstfiles=train_files, transform=transform)
val_dataset = H5Dataset(lstfiles=val_files, transform=transform)
test_dataset = H5Dataset(lstfiles=test_files, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = AttUNet_seg_std(in_channels=1, out_channels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# To store losses
history = {
    "train_loss": [],
    "val_loss": []
}

def plot_losses(history, epoch):
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss after {epoch} epochs')
    plt.legend()
    plt.savefig('imgs/'+modelname+'_loss.png')

# Training and validation loop
epochs = 50
for epoch in range(1, epochs + 1):
    # Training
    model.train()
    train_loss = 0
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    history["train_loss"].append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, target in val_loader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

    # Plot losses and test model every 10 epochs
    if epoch % 10 == 0:
        plot_losses(history, epoch)

        # Test the model
        test_outputs = []
        with torch.no_grad():
            for inputs, target in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_outputs.append(outputs.cpu().numpy())

        # TODO: np.save test outputs
        print(f"Test outputs saved for epoch {epoch}")
