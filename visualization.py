import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Set mode for testing
os.environ['MODE'] = 'test'
import config
import model  # model module
from test import load_test_dataset, load_checkpoint


def read_metrics(file_path):
    with open(file_path, 'r') as file:
        data_load = json.load(file)
    return data_load


def plot_metrics(metrics_plot, title):
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics_plot['avg_train_losses'], label='Avg Train Loss')
    plt.plot(metrics_plot['avg_val_losses'], label='Avg Validation Loss')
    plt.title(f'Average: {title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot SSIM scores
    plt.subplot(1, 2, 2)
    plt.plot(metrics_plot['avg_train_ssim_scores'], label='Avg Train SSIM')
    plt.plot(metrics_plot['avg_val_ssim_scores'], label='Avg Validation SSIM')
    plt.title(f'Average: {title} SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_orthoslices(ax, data, slice_idx):
    """
    Helper function to plot 3D data as orthoslices on given axes.
    """
    # Extracting the slices
    axial_slice = data[slice_idx[0], :, :]
    coronal_slice = data[:, slice_idx[1], :]
    sagittal_slice = data[:, :, slice_idx[2]]

    # Axial slice
    ax[0].imshow(axial_slice, cmap='gray')
    ax[0].set_title(f'Axial Slice {slice_idx[0]}')
    ax[0].axis('off')

    # Coronal slice
    ax[1].imshow(coronal_slice, cmap='gray')
    ax[1].set_title(f'Coronal Slice {slice_idx[1]}')
    ax[1].axis('off')

    # Sagittal slice
    ax[2].imshow(sagittal_slice, cmap='gray')
    ax[2].set_title(f'Sagittal Slice {slice_idx[2]}')
    ax[2].axis('off')


def visualize_sample(gt_visual, output_visual, title="Ground Truth vs Output", slice_idx=(84, 29, 29)):
    """
    Visualize 3D ground truth and output data as orthoslices.

    Args:
        gt_visual (torch.Tensor): The ground truth tensor.
        output_visual (torch.Tensor): The output tensor from the model.
        title (str): Title for the plot.
        slice_idx (tuple): Indices for axial, coronal, and sagittal slices.
    """
    gt_np = gt_visual.detach().cpu().numpy()  # Convert to numpy array
    output_np = output_visual.detach().cpu().numpy()  # Convert to numpy array

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle(title)

    # Plot Ground Truth Slices
    plot_orthoslices(axes[0], gt_np, slice_idx)

    # Plot Output Slices
    plot_orthoslices(axes[1], output_np, slice_idx)

    plt.show()


# Example usage
if __name__ == "__main__":
    # ------------- visualize some samples
    # Initialize model
    convLSTM_model = model.__dict__[config.d_arch_name](input_dim=config.input_dim,
                                                        hidden_dim=config.hidden_dim,
                                                        kernel_size=config.kernel_size,
                                                        num_layers=config.num_layers).to(config.device)
    # Load model checkpoint
    convLSTM_model = load_checkpoint(convLSTM_model, config.model_path)
    # Prepare test dataset
    test_loader = load_test_dataset()
    for data in test_loader:
        inputs = data['lr'].to(config.device)
        gt = data['gt'].to(config.device)

    # Generate output from the model
    convLSTM_model.eval()
    with torch.no_grad():
        output = convLSTM_model(inputs)
        sr = output[2]
    gt = gt.squeeze()
    sr = sr.squeeze()
    # Apply argmax along the class dimension (c)
    sr_3d = torch.argmax(sr, dim=1)
    print(sr_3d.shape)
    print(gt.shape)
    # Visualize the sample
    visualize_sample(gt, sr_3d, slice_idx=(84, 29, 29))

    # ------------- visualize the metrics
    # Directory where the results are stored
    results_dir = os.path.join("results", config.exp_name)
    num_folds = 5

    # Initialize lists to store aggregated metrics
    all_train_losses = []
    all_val_losses = []
    all_train_ssim_scores = []
    all_val_ssim_scores = []

    for fold in range(1, num_folds + 1):
        results_file = os.path.join(results_dir, f'_fold {fold}', 'training_metrics.json')
        if os.path.exists(results_file):
            metrics = read_metrics(results_file)
            all_train_losses.append(metrics['train_losses'])
            all_val_losses.append(metrics['val_losses'])
            all_train_ssim_scores.append(metrics['train_scores'])
            all_val_ssim_scores.append(metrics['val_scores'])
        else:
            print(f"Metrics file for fold {fold} not found.")

    # Calculate the average across all folds
    avg_metrics = {
        'avg_train_losses': np.mean(all_train_losses, axis=0),
        'avg_val_losses': np.mean(all_val_losses, axis=0),
        'avg_train_ssim_scores': np.mean(all_train_ssim_scores, axis=0),
        'avg_val_ssim_scores': np.mean(all_val_ssim_scores, axis=0)
    }

    plot_metrics(avg_metrics, "Training and Validation")
