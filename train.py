# Copyright 2023
# Xiaoyu Leo Yang
# Nanyang Technological University. All Rights Reserved.
# ==============================================================================
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, CPUPrefetcher, TrainValidImageDataset
from utils.utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter
from utils.criteria import SSIM3D

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    convLSTM_model, ema_convLSTM_model = build_model()
    print(f"Build `{config.g_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(convLSTM_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_d_model_weights_path:
        convLSTM_model = load_state_dict(convLSTM_model, config.pretrained_d_model_weights_path)
        print(f"Loaded `{config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume_d_model_weights_path:
        convLSTM_model, ema_convLSTM_model, start_epoch, optimizer, scheduler = load_state_dict(
            convLSTM_model,
            config.pretrained_d_model_weights_path,
            ema_convLSTM_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    ssim_model = SSIM3D()
    # Transfer the IQA model to the specified device
    ssim_model = ssim_model.to(device=config.device)

    best_ssim = 0

    for epoch in range(start_epoch, config.epochs):
        train(convLSTM_model,
              ema_convLSTM_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer)
        ssim = validate(convLSTM_model,
                        test_prefetcher,
                        epoch,
                        writer,
                        ssim_model,
                        "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_ssim": best_ssim,
                         "state_dict": convLSTM_model.state_dict(),
                         "ema_state_dict": ema_convLSTM_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        file_name=f"d_epoch_{epoch + 1}.pth.tar",
                        samples_dir=None,
                        results_dir=results_dir,
                        best_file_name="d_best.pth.tar",
                        last_file_name="d_last.pth.tar",
                        is_best=is_best,
                        is_last=is_last
                        )


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(image_dir=config.image_dir,
                                            label_dir=config.label_dir,
                                            mode=config.mode)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    # train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    train_prefetcher = CPUPrefetcher(train_dataloader)

    # reserve, to be modified after ....
    test_prefetcher = train_prefetcher

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    convLSTM_model = model.__dict__[config.g_arch_name](input_dim=config.input_dim,
                                                        hidden_dim=config.hidden_dim,
                                                        kernel_size=config.kernel_size,
                                                        output_size=config.output_size)

    convLSTM_model = convLSTM_model.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_convLSTM_model = AveragedModel(convLSTM_model, avg_fn=ema_avg)

    return convLSTM_model, ema_convLSTM_model


def define_loss() -> nn.MSELoss:
    criterion = nn.MSELoss()
    criterion = criterion.to(
        device=config.device)  # Assuming 'config.device' is defined and specifies the device (e.g., 'cuda' or 'cpu')

    return criterion


def define_optimizer(model_train) -> optim.Adam:
    optimizer = optim.Adam(model_train.parameters(),
                           config.model_lr,
                           config.model_betas,
                           config.model_eps,
                           config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer,
                                    config.lr_scheduler_milestones,
                                    config.lr_scheduler_gamma)

    return scheduler


def train(
        train_model: nn.Module,
        ema_convLSTM_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    train_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)

        # Initialize generator gradients
        train_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = train_model(lr)
            loss = criterion(sr, gt)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_convLSTM_model.update_parameters(train_model)

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        rrdbnet_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        ssim_3d: any,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    rrdbnet_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = rrdbnet_model(lr)

            # Statistical loss value for terminal data output
            ssim = ssim_3d(sr, gt)
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return ssimes.avg


if __name__ == "__main__":
    main()