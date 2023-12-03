import json
import os
import matplotlib.pyplot as plt
import numpy as np
import config


def read_metrics(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def plot_metrics(metrics, title):
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['avg_train_losses'], label='Avg Train Loss')
    plt.plot(metrics['avg_val_losses'], label='Avg Validation Loss')
    plt.title(f'Average: {title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot SSIM scores
    plt.subplot(1, 2, 2)
    plt.plot(metrics['avg_train_ssim_scores'], label='Avg Train SSIM')
    plt.plot(metrics['avg_val_ssim_scores'], label='Avg Validation SSIM')
    plt.title(f'Average: {title} SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


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
        all_train_ssim_scores.append(metrics['train_ssim_scores'])
        all_val_ssim_scores.append(metrics['val_ssim_scores'])
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
