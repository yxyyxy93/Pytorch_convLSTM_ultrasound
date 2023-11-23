import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import ConvLSTM3DClassifier
from utlis import read_csv_to_3d_array

# **************** Data Preparation
# Example of loading input data and label data
input_data = read_csv_to_3d_array('test_woven_[#090]_8/base_model_shiftseed_3/_10m_20231116/_snr_32.00.csv')
label_data = read_csv_to_3d_array('test_woven_[#090]_8/base_model_shiftseed_3/structure_20231116.csv')

num_epochs = 10

# Split your data into training and validation sets
# Adjust the splitting logic as per your dataset size and requirements
train_input, val_input = input_data[:int(len(input_data) * 0.8)], input_data[int(len(input_data) * 0.8):]
train_labels, val_labels = label_data[:int(len(label_data) * 0.8)], label_data[int(len(label_data) * 0.8):]

# Convert to PyTorch tensors
train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
val_input_tensor = torch.tensor(val_input, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

# Add a batch dimension to each tensor
train_input_tensor = train_input_tensor.unsqueeze(0)
train_labels_tensor = train_labels_tensor.unsqueeze(0)
val_input_tensor = val_input_tensor.unsqueeze(0)
val_labels_tensor = val_labels_tensor.unsqueeze(0)

# Now the shapes should be (1, 21, 21, 400) or similar, depending on your data
print(train_input_tensor.shape)
print(train_labels_tensor.shape)

# Create DataLoaders
train_dataset = TensorDataset(train_input_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_input_tensor, val_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# **************************** Model Initialization, Loss, and Optimizer
input_dim = 21  # Adjust according to your data
hidden_dim = 50  # Example value, you may need to tweak this
kernel_size = (3, 3)  # Example kernel size
output_size = train_labels.shape  # Specify your output size

model = ConvLSTM3DClassifier(input_dim, hidden_dim, kernel_size, output_size)

criterion = nn.MSELoss()  # or an appropriate 3D loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# *************************** Training Loop
for epoch in range(num_epochs):
    for data, labels in train_loader:
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_labels)
            # Perform evaluation, e.g., calculating accuracy or other metrics
