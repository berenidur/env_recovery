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

std_const=torch.pi/2/np.sqrt(6)

modelname='attunet_std'

# Define a dummy dataset
class PuntomatsDS(Dataset):
    def __init__(self, lstfiles, transform=None):
        self.lstfiles = lstfiles
        self.transform = transform

    def __len__(self):
        return len(self.lstfiles)

    def __getitem__(self, idx):
        mat = scipy.io.loadmat(self.lstfiles[idx])
        comp_env = mat["comp_env"][:1200,:]
        a_0 = mat["a_0"]
        b_0 = mat["b_0"]
        target_std = a_0*std_const

        if self.transform:
            comp_env = self.transform(comp_env)
            target_std = self.transform(target_std)
        return comp_env, std_const
    
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
train_dataset = PuntomatsDS(lstfiles=train_files, transform=transform)
val_dataset = PuntomatsDS(lstfiles=val_files, transform=transform)
test_dataset = PuntomatsDS(lstfiles=test_files, transform=transform)

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
