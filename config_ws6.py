# Copyright 2023
# Xiaoyu Leo Yang
# Nanyang Technological university
# ==============================================================================
import random
import numpy as np
import torch
import os


def show_cuda_gpu_info():
    print("Checking CUDA and GPU status...")

    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device("cuda", 0)
        torch.backends.cudnn.benchmark = True
        print(f"Number of GPUs Available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    return device


# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = show_cuda_gpu_info()

# Model arch config
input_dim = 1
hidden_dim = 1
kernel_size = (3, 3)
output_size = (50, 50, 168)  #

# Experiment name, easy to save weights and log files
exp_name = "convLSTM_baseline"

d_arch_name = "ConvLSTM3DClassifier"

# How many iterations to print the training result
train_print_frequency = 5
valid_print_frequency = 5

# Initialize mode as None
mode = None
mode = os.getenv('MODE', 'train')  # Default to 'train' if not set

if mode == "train":
    # Dataset address
    image_dir = '/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_[#45n45#090]_4_defect/sim_data'  # path to the 'sim_data' directory
    label_dir = '/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_[#45n45#090]_4_defect/sim_struct'  # path to the 'sim_struct' directory

    batch_size = 2
    num_workers = 2

    # The address to load the pretrained model
    pretrained_model_path = "./results/pretrained_models/ConvLSTM_pretrain.pth.tar"

    # The address to load the pretrained model
    pretrained_d_model_weights_path = ""

    # Incremental training and migration training
    resume_d_model_weights_path = f""

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 100

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
    # Test data address To be modified ...
    image_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_data'  # path to the 'sim_data' directory
    label_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_struct'  # path to the 'sim_struct' directory

    model_path = r"D:\python_work\ConvLSTM_3dultrasound\results\convLSTM_baseline\d_best.pth.tar"
