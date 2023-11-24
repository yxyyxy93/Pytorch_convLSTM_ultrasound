# Copyright 2023
# Xiaoyu Leo Yang
# Nanyang Technological university
# ==============================================================================
import random
import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# # Use GPU for training by default
# device = torch.device("cuda", 0)
# # Turning on when the image size does not change during training can speed up training
# cudnn.benchmark = True

# Use CPU for training
device = torch.device("cpu")

# Model arch config
input_dim = 1
hidden_dim = 1
kernel_size = (3, 3)
output_size = (50, 50, 168)  #

# Current configuration parameter method
mode = "train"
# mode = "test"

# Experiment name, easy to save weights and log files
exp_name = "convLSTM_baseline"

g_arch_name = "ConvLSTM3DClassifier"

# How many iterations to print the training result
train_print_frequency = 5
valid_print_frequency = 5

if mode == "train":
    # Dataset address
    image_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_data'  # path to the 'sim_data' directory
    label_dir = r'D:\python_work\ConvLSTM_3dultrasound\dataset\sim_struct'  # path to the 'sim_struct' directory

    batch_size = 2
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
    epochs = 100

    # Optimizer parameter
    model_lr = 1e-5
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.5
    # How many iterations to print the training result
    print_frequency = 200

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

if mode == "test":
    # Test data address To be modified ...
    lr_dir = f"./data/ImageTest/noisy_images"
    sr_dir = f"./results/test/{exp_name}"
    hr_dir = f"./data/ImageTest/origin_images"

    # model_path = r"F:\Xiayang\python_work\SRGAN-PyTorch-ultrasonic\samples\SRResNet_baseline/g_epoch_100.pth.tar"
    # model_path = r".\results\ESRGAN_x2\g_best.pth.tar"
    # model_path = r"F:\Xiayang\python_work\SRGAN-PyTorch-ultrasonic\samples\SRGan_baseline\g_epoch_100.pth.tar"
