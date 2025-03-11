import torch
import numpy as np
import pickle
import os

import model_autoencoders
from utils import *

# Model parameters
modelname = 'autoencoder_breast_v0.1'
checkpoint_path = f'models/{modelname}/latest.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
with open('breast_data_arrays_CNN.pkl', 'rb') as f:
    data_splits = pickle.load(f)

val_files = data_splits['val_windows']

class H5Dataset_windows(torch.utils.data.Dataset):
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
        # y_beta = self.windows[idx].beta
        # y_k = self.windows[idx].k
        
        x = np.array([x_R, x_S], dtype=np.float32)
        y = np.array([y_R, y_S], dtype=np.float32)
        # y = np.array([y_R, y_S, y_beta, y_k], dtype=np.float32)
        
        x = np.expand_dims(x, axis=0)  # Add channel dimension
        y = np.expand_dims(y, axis=0)  # For consistency
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y

def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

val_dataset = H5Dataset_windows(val_files, transform=to_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)

# Load model
ModelClass = getattr(model_autoencoders, modelname.replace(".", "_"))
model = ModelClass().to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model checkpoint loaded.")
else:
    raise FileNotFoundError("Checkpoint not found! Make sure the model is trained and saved.")

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

# Convert to arrays
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# Save results
np.save(f'models/{modelname}/validation_predictions.npy', predictions)
np.save(f'models/{modelname}/validation_actuals.npy', actuals)
print("Validation outputs saved.")