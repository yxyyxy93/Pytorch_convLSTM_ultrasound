import numpy as np


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
