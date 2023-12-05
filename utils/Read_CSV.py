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


# if __name__ == "__main__":
#     import os
#     import re
#
#     # Define the root directory to start the search
#     root_directory = r'D:\python_work\ConvLSTM_3dultrasound\dataset'
#     # Define the destination directories
#     image_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_data_npy'  # path to the 'sim_data' directory
#     label_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_struct_npy'  # path to the 'sim_struct' directory
#     exp_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\exp_data_npy'  # path to the 'exp_data' directory
#
#     # Create the destination directories if they don't exist
#     os.makedirs(image_dir, exist_ok=True)
#     os.makedirs(label_dir, exist_ok=True)
#     os.makedirs(exp_dir, exist_ok=True)
#
#     # Counter for subfolders
#     subfolder_counter = 0
#
#     # Loop through all subdirectories and files
#     for subdir, dirs, files in os.walk(root_directory):
#         subfolder_name = os.path.basename(subdir)
#         # Match subfolder names ending with 'seed_' followed by a number
#         match = re.search(r'seed_(\d+)$', subfolder_name)
#         if match:
#             # Extract the number after 'seed_'
#             subfolder_number = match.group(1)
#         else:
#             subfolder_number = "experiment"  # this is maybe an exp.. data
#
#         if subfolder_number is None:
#             continue  # Skip subfolders without a trailing number
#
#         for file_counter, file in enumerate(files, start=0):
#             # Check if the file is a .csv file
#             if file.endswith('.csv'):
#                 file_path = os.path.join(subdir, file)
#                 # Read the .csv file
#                 array3d = read_csv_to_3d_array(file_path)
#                 # Count NaN values
#                 nan_count = np.isnan(array3d).sum()
#                 if nan_count > 0:
#                     print(f"File: {file_path} has {nan_count} NaN values")
#             else:
#                 continue
#
#             # Determine the destination directory based on the file name
#             if file.startswith('_snr'):
#                 destination_directory = image_dir
#             elif file.startswith('structure'):
#                 destination_directory = label_dir
#             elif file.startswith('exp'):
#                 destination_directory = exp_dir
#                 subfolder_number = file[0:-4]  # keep original name for exp.
#             else:
#                 continue  # Skip files that do not match the criteria
#
#             # Construct the new filename
#             new_filename = f"{subfolder_number}-{file_counter}.npy"
#             new_file_path = os.path.join(destination_directory, new_filename)
#
#             # print(new_file_path)
#             # Save the array to a .npy file
#             np.save(new_file_path, array3d)
#             print(f"Array saved to {new_file_path}")
