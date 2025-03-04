import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cnn_model import CNNModel
from utils import *

# Define model and device
modelname = 'cnn_v0.1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

# Load dataset
class H5Dataset_windows(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.expand_dims(x, axis=0)  # Add channel dimension
        y = np.expand_dims(y, axis=0)  # For consistency
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

# Load dataset splits
with open('data_arrays_CNN.pkl', 'rb') as f:
    data_splits = pickle.load(f)

val_files_x, val_files_y = data_splits['val_comp_env_windows'], data_splits['val_validRS_values']
val_dataset = H5Dataset_windows(val_files_x, val_files_y, transform=to_tensor)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Load checkpoint
epoch = 40  # Change this to the desired epoch
checkpoint_path = f'models/{modelname}/epoch_{epoch}.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Compute outputs
outputs_list = []
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs_list.append(outputs.cpu().numpy())

# Convert lists to arrays
outputs_array = np.concatenate(outputs_list, axis=0)

# Save results
checkpath(f'outputs/{modelname}/')
np.save(f'outputs/{modelname}/epoch_{epoch}.npy', outputs_array)

print(f"Outputs saved for epoch {epoch}.")
