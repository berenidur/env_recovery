import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset

import custom_models
from utils import *

# Model parameters
v=0.1
modelname = f'customnetwork_v{v}'
datasetname = f'H5Dataset_windows_custom_v{v}'
epoch = 800
checkpoint_path = f'models/{modelname}/{"latest" if epoch==0 else f"epoch_{epoch}"}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
with open('breast_noNaN_data_arrays_CNN.pkl', 'rb') as f:
    data_splits = pickle.load(f)

val_files = data_splits['val_windows'][:20]

def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

datasetClass = getattr(custom_models, datasetname.replace(".", "_"))  # Get the class
val_dataset = datasetClass(val_files, transform=to_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)

# Load model
ModelClass = getattr(custom_models, modelname.replace(".", "_"))
model = ModelClass(latent_dim=3).to(device) 

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model epoch {"latest" if epoch==0 else epoch} loaded.')
else:
    raise FileNotFoundError(f'Epoch {"latest" if epoch==0 else epoch} not found!')

model.eval()
predictions = []
actuals = []
predicted_a0s = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predicted_a0, outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())
        predicted_a0s.append(predicted_a0.cpu().numpy())

# Convert to arrays
actuals = np.squeeze(actuals)
predicted_a0s = np.squeeze(predicted_a0s)
predictions = np.squeeze(predictions)
predictions=np.hstack((predicted_a0s[:,None], actuals))


# Save results
np.save(f'models/{modelname}/validation_predictions.npy', predictions)
np.save(f'models/{modelname}/validation_actuals.npy', actuals)
print("Validation outputs saved.")