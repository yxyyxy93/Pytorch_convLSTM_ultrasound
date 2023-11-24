import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to open the file dialog and return the selected file path
def select_file():
    # Create a root window, but keep it hidden
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog and store the selected file path
    file_path = filedialog.askopenfilename()

    # Destroy the root window
    root.destroy()

    return file_path


def read_csv_to_3d_array(filepath):
    # Open the file
    with open(filepath, 'r') as file:
        # Read the first line to get the dimensions
        x, y, z = map(int, file.readline().strip().split(','))

        # Initialize an empty 3D NumPy array
        data_3d_np = np.zeros((x, y, z))

        # Read the rest of the lines and fill the 3D array
        for i in range(x):
            for j in range(y):
                line = file.readline().strip().split(',')
                data_3d_np[i, j, :] = np.array(line[0:z], dtype=float)

    return data_3d_np


def plot_slices(data):
    x, y, z = data.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting a slice along each dimension
    axes[0].imshow(data[x // 2, :, :], cmap='viridis', aspect='auto')
    axes[0].set_title('Slice along X-axis')
    axes[0].set_xlabel('Z')
    axes[0].set_ylabel('Y')

    axes[1].imshow(data[:, y // 2, :], cmap='viridis', aspect='auto')
    axes[1].set_title('Slice along Y-axis')
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')

    axes[2].imshow(data[:, :, z // 2], cmap='viridis', aspect='auto')
    axes[2].set_title('Slice along Z-axis')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('X')

    plt.tight_layout()
    plt.show()


# Call the function and get the selected file path
selected_file_path = select_file()
# Check if a file was selected
if selected_file_path:
    print(f"Selected file: {selected_file_path}")
    # You can now use selected_file_path to read the CSV file
    # ... Your code to read the CSV file ...
else:
    print("No file was selected.")

# # Replace with the path to your CSV file
# file_path = 'path/to/your/csvfile.csv'

# Read the CSV and convert it to a 3D array
data_3d = read_csv_to_3d_array(selected_file_path)

# Output
print("Size of data_3d:", data_3d.shape)
plot_slices(data_3d)
