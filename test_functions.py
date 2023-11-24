import torch
from torch.utils.data import DataLoader
from dataset import \
    TrainValidImageDataset  # Make sure 'dataset.py' is in the same directory or adjust the import path accordingly

# Parameters for the dataset and dataloader
image_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_data'  # path to the 'sim_data' directory
label_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_struct'  # path to the 'sim_struct' directory
time_steps = 400  # Replace with your actual image size
channels = 1
H = 21
W = 21
batch_size = 4  # Define the batch size for the dataloader
mode = 'train'  # Or 'valid', depending on what mode you want to test

# ... (previous code for importing and dataset parameters)

# Initialize the dataset
dataset = TrainValidImageDataset(image_dir=image_dir, label_dir=label_dir, mode=mode)

# Retrieve and check one sample from the dataset
sample = dataset.__getitem__(1)
print("Sample 'lr' shape:", sample['lr'].shape)
print("Sample 'gt' shape:", sample['gt'].shape)

# Initialize the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test loop - iterate over the dataloader
for i, batch in enumerate(dataloader):
    gt, lr = batch['gt'], batch['lr']
    print(f"Batch {i}:")
    print("  'lr' shape:", lr.shape)
    print("  'gt' shape:", gt.shape)

    # Uncomment the following line if you are sure about the expected shape
    # assert gt.shape == (batch_size, time_steps, channels, H, W)
    assert lr.shape == (batch_size, time_steps, channels, H, W)

    # Check if the ground truth and data are associated correctly
    # This depends on how your dataset associates 'lr' and 'gt'.
    # Add any specific checks here based on your dataset structure.

    if i == 2:  # Check a few batches for testing purposes
        break