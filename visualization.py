import json

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch


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
    plt.plot(metrics_plot['avg_train_scores'], label='Avg Train SSIM')
    plt.plot(metrics_plot['avg_val_scores'], label='Avg Validation SSIM')
    plt.title(f'Average: {title} SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_orthoslices(data, slice_idx):
    """
    Helper function to plot 3D data as orthoslices on given axes.
    """
    if data.ndim == 3:  # For 3D data
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))

        def update(val):
            slice_index = int(slider.val)
            xy_slice = data[slice_index, :, :].squeeze()
            ax[0].clear()
            ax[0].imshow(xy_slice, cmap='gray')
            ax[0].set_title('XY slice')
            # fig.canvas.draw_idle()

        depth, height, width = data.shape[0], data.shape[1], data.shape[2]
        # Extracting the slices
        axial_slice = data[slice_idx[0], :, :]
        coronal_slice = data[:, slice_idx[1], :]
        sagittal_slice = data[:, :, slice_idx[2]]

        # Axial slice
        ax[0].imshow(axial_slice, cmap='gray', aspect='auto')
        ax[0].set_title(f'Axial Slice {slice_idx[0]}')
        ax[0].axis('off')

        # Coronal slice
        ax[1].imshow(coronal_slice, cmap='gray', aspect='auto')
        ax[1].set_title(f'Coronal Slice {slice_idx[1]}')
        ax[1].axis('off')

        # Sagittal slice
        ax[2].imshow(sagittal_slice, cmap='gray', aspect='auto')
        ax[2].set_title(f'Sagittal Slice {slice_idx[2]}')
        ax[2].axis('off')

        # Create the slider
        ax_slider = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Slice', 0, depth - 1, valinit=depth // 2, valfmt='%0.0f')

        slider.on_changed(update)

    elif data.ndim == 2:  # For 2D data
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(data, cmap='gray', aspect='auto')
        ax.axis('off')
    else:
        raise ValueError("Data must be either 2D or 3D")

    plt.show()


def visualize_sample(gt_visual, output_visual, slice_idx=(84, 29, 29)):
    """
    Visualize 3D ground truth and output data as orthoslices.

    Args:
        gt_visual (torch.Tensor): The ground truth tensor.
        output_visual (torch.Tensor): The output tensor from the model.
        slice_idx (tuple): Indices for axial, coronal, and sagittal slices.
    """
    gt_np = gt_visual.detach().cpu().numpy()  # Convert to numpy array
    output_np = output_visual.detach().cpu().numpy()  # Convert to numpy array

    # Display statistics for Ground Truth
    print("Ground Truth Statistics:")
    print(f"Max: {gt_np.max()}, Min: {gt_np.min()}, Mean: {gt_np.mean()}, Std: {gt_np.std()}")
    # Display statistics for Output
    print("Output Statistics:")
    print(f"Max: {output_np.max()}, Min: {output_np.min()}, Mean: {output_np.mean()}, Std: {output_np.std()}")

    # Plot Output Slices
    plot_orthoslices(output_np, slice_idx)

    # Plot Ground Truth Slices
    plot_orthoslices(gt_np, slice_idx)


if __name__ == "__main__":
    import numpy as np
    import os
    import model

    # Set mode for testing
    os.environ['MODE'] = 'train'
    import config
    from test import load_test_dataset, load_checkpoint

    # ------------- visualize some samples
    # Initialize model
    convLSTMmodel = model.__dict__[config.d_arch_name](input_dim=config.input_dim,
                                                       hidden_dim=config.hidden_dim,
                                                       new_channel=config.output_dim,
                                                       new_seq_len=config.output_tl,
                                                       kernel_size=config.kernel_size,
                                                       num_layers=config.num_layers,
                                                       batch_first=True).to(config.device)

    model = convLSTMmodel.to(device=config.device)

    results_dir = config.results_dir
    fold_number = 5  # Change as needed
    model_filename = "d_best.pth.tar"
    model_path = os.path.join(results_dir, f"_fold {fold_number}", model_filename)
    # Load model checkpoint
    convLSTM_model = load_checkpoint(model, model_path)

    # Prepare test dataset
    test_loader = load_test_dataset()
    for data in test_loader:
        inputs = data['lr'].to(config.device)
        gt = data['gt'].to(config.device)

    # Generate output from the model
    convLSTM_model.eval()
    with torch.no_grad():
        output = convLSTM_model(inputs)

    # Visualize the sample
    visualize_sample(output[:, 0].squeeze(), data['loc_xy'].squeeze(), slice_idx=(128, 8, 8))
    visualize_sample(output[:, 1].squeeze(), gt[:, 1].squeeze(), slice_idx=(128, 8, 8))

    # ------------- visualize the metrics
    # Directory where the results are stored
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
        'avg_train_scores': np.mean(all_train_ssim_scores, axis=0),
        'avg_val_scores': np.mean(all_val_ssim_scores, axis=0)
    }

    plot_metrics(avg_metrics, "Training and Validation")
