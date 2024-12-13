import os
import shutil

def flatten_directory(folder_path):
    """
    Move all files from subfolders into the root folder and delete empty subfolders.

    Args:
        folder_path (str): The root folder to flatten.
    """
    for root, dirs, files in os.walk(folder_path, topdown=False):  # Traverse bottom-up to handle files first
        for file in files:
            # Move file to the root folder
            file_path = os.path.join(root, file)
            new_path = os.path.join(folder_path, file)
            
            # Handle duplicate filenames
            if os.path.exists(new_path):
                base, ext = os.path.splitext(file)
                counter = 2
                while os.path.exists(new_path):
                    new_path = os.path.join(folder_path, f"{base}_{counter}{ext}")
                    counter += 1
            
            shutil.move(file_path, new_path)
            print(f"Moved: {file_path} -> {new_path}")
        
        # Remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)  # Only deletes if empty
                print(f"Deleted empty folder: {dir_path}")
            except OSError as e:
                print(f"Error deleting folder {dir_path}: {e}")

# Usage example
folder_to_flatten = './breast'
flatten_directory(folder_to_flatten)
