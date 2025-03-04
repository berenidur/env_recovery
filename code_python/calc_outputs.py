import torch
import numpy as np
import pickle
import os

from scipy.ndimage import binary_dilation

from unets import UNet
from utils import *

modelname = 'unet_v0.1.2_imdilation'
h5_path = '../data/dataoncosalud/res_valid/comp_env_data.h5'
dataset = 'comp_env_interp_1'
n = 57  # H,W of each window
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load data splits
with open('data_splits.pkl', 'rb') as f:
    data_splits = pickle.load(f)

test_files = data_splits['test_files']
test_files.sort()
Q1 = data_splits['Q1']

imdilate_structure=np.ones((n,n))

# Define dataset class
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, lstgroups, dataset, Q1, transform=None):
        self.lstgroups = lstgroups
        self.dataset = dataset
        self.transform = transform
        self.Q1 = Q1

    def __len__(self):
        return len(self.lstgroups)

    def __getitem__(self, idx):
        x = load_h5_dataset(h5_path, self.lstgroups[idx], self.dataset)[:self.Q1, :] / 255
        y = load_h5_dataset(h5_path, self.lstgroups[idx], 'validRS')[:self.Q1, :]
        y[-(n - 1) // 2 :, :] = 0  # Set last rows to 0

        if x.shape[1] == 512:
            x = x[:, ::2]
            y = y[:, ::2]

        y = binary_dilation(y,structure=imdilate_structure).astype(y.dtype)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

# Convert to tensor
def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

# Create test dataset and DataLoader
transform = to_tensor
test_dataset = H5Dataset(lstgroups=test_files, dataset=dataset, Q1=Q1, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loop through saved models at epochs 5, 10, ..., 60
for epoch in range(25, 26, 5):
    checkpoint_path = f'models/{modelname}_epoch_{epoch}.pth'

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        continue

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Processing test files for epoch {epoch}...")

    # Run inference on test data
    test_outputs = []
    with torch.no_grad():
        for inputs, _ in test_loader:  # We don't need targets
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_outputs.append(outputs.cpu().numpy())

    # Save test outputs
    test_outputs = np.concatenate(test_outputs, axis=0)
    np.save(f'outputs/{modelname}_test_outputs_epoch_{epoch}.npy', test_outputs)
    print(f"Test outputs saved for epoch {epoch}.")

print("Processing complete.")
