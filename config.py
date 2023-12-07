# Copyright 2023
# Xiaoyu Leo Yang
# Nanyang Technological university
# ==============================================================================
import random
import numpy as np
import torch
import os


def show_cuda_gpu_info():
    if torch.cuda.is_available():
        print("CUDA is available.")
        device_use = torch.device("cuda", 2)
        torch.backends.cudnn.benchmark = True
        print(f"Number of GPUs Available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
        device_use = torch.device("cpu")

    return device_use


# from torch.backends import cudnn
# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = show_cuda_gpu_info()

# Model arch config
input_dim = 1
hidden_dim = 64
kernel_size = (3, 3)
output_dim = 2  # 2 or more classes
output_tl = 168  # the depth length
num_layers = 2

# ------------- choose from models
d_arch_name = "ConvLSTM"

# ---------- choose from loss functions
loss_function = "MulticlassDiceLoss"  # Options: myCrossEntropyLoss, MulticlassDiceLoss, etc.
val_function = "PixelAccuracy"

# Experiment name, easy to save weights and log files
exp_name = d_arch_name + "_baseline"

# How many iterations to print the training result
train_print_frequency = 2
valid_print_frequency = 10

# Initialize mode as None
# mode = os.getenv('MODE', 'train')  # Default to 'train' if not set
mode = os.environ.get('MODE')

if mode == "train":
    print("train mode")
    # Dataset address
    image_dir = r'.\dataset\sim_data'  # path to the 'sim_data' directory
    label_dir = r'.\dataset\sim_struct'  # path to the 'sim_struct' directory

    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_path = "./results/pretrained_models/ConvLSTM_pretrain.pth.tar"

    # The address to load the pretrained model
    pretrained_d_model_weights_path = ""
    # Incremental training and migration training
    resume_d_model_weights_path = f""
    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 50

    # Optimizer parameter
    model_lr = 1e-3
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 1e-5  # weight decay (e.g., 1e-4 or 1e-5) can be beneficial as it adds L2 regularization.

    # EMA parameter
    model_ema_decay = 0.5
    # How many iterations to print the training result
    print_frequency = 100

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.1
elif mode == "test":
    print("testing mode")
    # Test data address To be modified ...
    image_dir = r'.\dataset\sim_data'  # path to the 'sim_data' directory
    label_dir = r'.\dataset\sim_struct'  # path to the 'sim_struct' directory

    # Constructing the path
    results_dir = os.path.join("results", f"{exp_name}_2023-12-07")
