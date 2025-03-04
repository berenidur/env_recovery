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

def plot_losses(history, epoch, modelname, start_epoch=1):
    plt.figure(figsize=(8, 6))
    plt.plot(range(start_epoch, epoch + 1), history['train_loss'][start_epoch - 1:], label='Train Loss')
    plt.plot(range(start_epoch, epoch + 1), history['val_loss'][start_epoch - 1:], label='Validation Loss')
    plt.grid(True)
    plt.xlim(start_epoch, epoch)  # Set x-axis limits
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss from epoch {start_epoch} to {epoch}')
    plt.legend()
    plt.savefig(f'imgs/{modelname}_loss.png')


def calculate_total_windows_h5(h5_path, n):
    total_windows = 0
    with h5py.File(h5_path, 'r') as file:
        for group in file.keys():
            dataset_name = f'{group}/comp_env_interp_1'
            if dataset_name in file:
                data = np.array(file[dataset_name])
                if data.ndim == 2:
                    num_windows = ((data.shape[0] - n + 1) * (data.shape[1] - n + 1))
                    total_windows += num_windows
                else:
                    raise ValueError(f"Dataset {dataset_name} is not 2-dimensional")
            else:
                raise ValueError(f"Dataset {dataset_name} not found")
    return total_windows

def get_window_xy_h5(h5_path, n, window_idx):
    total_windows = calculate_total_windows_h5(h5_path, n)
    if window_idx >= total_windows:
        raise IndexError("Window index out of range")
    
    current_window = 0
    with h5py.File(h5_path, 'r') as file:
        for group in file.keys():
            dataset_name = f'{group}/comp_env_interp_1'
            valid_rs_name = f'{group}/validRS'
            if dataset_name in file and valid_rs_name in file:
                data = np.array(file[dataset_name])
                if data.ndim == 2:
                    num_windows = ((data.shape[0] - n + 1) * (data.shape[1] - n + 1))
                    if current_window + num_windows > window_idx:
                        local_idx = window_idx - current_window
                        row_idx = local_idx // (data.shape[1] - n + 1)
                        col_idx = local_idx % (data.shape[1] - n + 1)
                        comp_env_window = data[row_idx:row_idx+n, col_idx:col_idx+n]
                        
                        valid_rs = np.array(file[valid_rs_name])
                        valid_rs_value = valid_rs[row_idx, col_idx]

                        return comp_env_window, valid_rs_value
                    current_window += num_windows
                else:
                    raise ValueError(f"Dataset {dataset_name} is not 2-dimensional")
            else:
                raise ValueError(f"Dataset {dataset_name} or {valid_rs_name} not found")
    raise IndexError("Window index out of range")