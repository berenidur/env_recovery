import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def checkpath(p):
    if not os.path.exists(p):
        os.makedirs(p)

def load_h5_dataset(h5_path, group, dataset):
    with h5py.File(h5_path, 'r') as file:
        data = np.array(file[group][dataset])
    return data

def disp_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365)
    result = []
    if years > 0:
        result.append(f"{years} years")
    if days > 0:
        result.append(f"{days} days")
    if hours > 0:
        result.append(f"{hours} hours")
    if minutes > 0:
        result.append(f"{minutes} minutes")
    if seconds > 0:
        result.append(f"{seconds:.3f} seconds")
    return ", ".join(result)

def plot_losses(history, epoch, modelname):
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss after {epoch} epochs')
    plt.legend()
    plt.savefig('imgs/'+modelname+'_loss.png')