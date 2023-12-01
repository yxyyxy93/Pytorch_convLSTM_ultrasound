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
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, CPUPrefetcher, TrainValidImageDataset, show_dataset_info
from utils.utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter
from utils.criteria import SSIM3D
import json
import datetime

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")
    # after the DataLoader is initialized
    show_dataset_info(train_prefetcher, show_sample_slices=True)
    convLSTM_model, ema_convLSTM_model = build_model()
    print(f"Build `{config.d_arch_name}` model successfully.")
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
    samples_dir = os.path.join("../samples", config.exp_name)
    results_dir = os.path.join("../results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("../samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    ssim_model = SSIM3D()
    # Transfer the IQA model to the specified device
    ssim_model = ssim_model.to(device=config.device)

    # Initialize lists to store metrics for each epoch
    best_ssim = 0
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_ssim_scores = []
    epoch_val_ssim_scores = []
    # Get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    for epoch in range(start_epoch, config.epochs):
        avg_train_loss, avg_train_ssim = train(convLSTM_model,
                                               ema_convLSTM_model,
                                               train_prefetcher,
                                               criterion,
                                               optimizer,
                                               epoch,
                                               scaler,
                                               writer,
                                               ssim_model)  # Pass the SSIM model to train
        avg_val_loss, avg_val_ssim = validate(convLSTM_model,
                                              test_prefetcher,
                                              epoch,
                                              writer,
                                              criterion,  # Pass the loss criterion to validate
                                              ssim_model,
                                              "Test")

        # After train and validate calls
        # Save the training and validation metrics
        epoch_train_losses.append(avg_train_loss)
        epoch_train_ssim_scores.append(avg_train_ssim)
        epoch_val_losses.append(avg_val_loss)
        epoch_val_ssim_scores.append(avg_val_ssim)

        metrics = {
            "train_losses": epoch_train_losses,
            "train_ssim_scores": epoch_train_ssim_scores,
            "val_losses": epoch_val_losses,
            "val_ssim_scores": epoch_val_ssim_scores
        }
        # Save to a JSON file
        results_file = os.path.join(results_dir, f'training_metrics_{current_date}.json')
        with open(results_file, 'w') as f:
            json.dump(metrics, f)
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = avg_val_ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_ssim = max(avg_val_ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_ssim": best_ssim,
                         "state_dict": convLSTM_model.state_dict(),
                         "ema_state_dict": ema_convLSTM_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        file_name=f"d_epoch_{epoch + 1}.pth.tar",
                        samples_dir="",
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
    convLSTM_model = model.__dict__[config.d_arch_name](input_dim=config.input_dim,
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


def define_scheduler(optimizer) -> MultiStepLR:
    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                         milestones=config.lr_scheduler_milestones,
                                         gamma=config.lr_scheduler_gamma)

    return scheduler


def train(
        train_model: nn.Module,
        ema_convLSTM_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        ssim_3d: any  # Add the SSIM computation function
) -> (float, float):  # Change return type to include both loss and SSIM
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    ssimes = AverageMeter("SSIM", ":6.6f")  # New meter for SSIM
    progress = ProgressMeter(batches, [batch_time, data_time, losses, ssimes], prefix=f"Epoch: [{epoch + 1}]")

    train_model.train()
    batch_index = 0
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    end = time.time()

    while batch_data is not None:
        data_time.update(time.time() - end)
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)
        train_model.zero_grad(set_to_none=True)

        with amp.autocast():
            sr = train_model(lr)
            loss = criterion(sr, gt)
            ssim = ssim_3d(sr, gt)  # Compute SSIM

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_convLSTM_model.update_parameters(train_model)

        losses.update(loss.item(), lr.size(0))
        ssimes.update(ssim.item(), lr.size(0))  # Update SSIM meter

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config.train_print_frequency == 0:
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/SSIM", ssim.item(), batch_index + epoch * batches + 1)  # Log SSIM
            progress.display(batch_index + 1)

        batch_data = train_prefetcher.next()
        batch_index += 1

    avg_loss = losses.avg
    avg_ssim = ssimes.avg  # Calculate average SSIM
    return avg_loss, avg_ssim  # Return both average loss and SSIM


def validate(
        validate_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        criterion: nn.MSELoss,  # Add criterion for loss computation
        ssim_3d: any,
        mode: str
) -> (float, float):  # Change return type to include both loss and SSIM
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")  # New meter for loss
    ssimes = AverageMeter("SSIM", ":6.6f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, losses, ssimes], prefix=f"{mode}: ")

    validate_model.eval()
    batch_index = 0
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            with amp.autocast():
                sr = validate_model(lr)
                loss = criterion(sr, gt)  # Compute loss
                ssim = ssim_3d(sr, gt)  # Compute SSIM

            losses.update(loss.item(), lr.size(0))  # Update loss meter
            ssimes.update(ssim.item(), lr.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % config.valid_print_frequency == 0:
                writer.add_scalar(f"{mode}/Loss", loss.item(), epoch + 1)  # Log loss
                writer.add_scalar(f"{mode}/SSIM", ssim.item(), epoch + 1)
                progress.display(batch_index + 1)

            batch_data = data_prefetcher.next()
            batch_index += 1

    progress.display_summary()
    avg_loss = losses.avg
    avg_ssim = ssimes.avg  # Calculate average SSIM
    return avg_loss, avg_ssim  # Return both average loss and SSIM


if __name__ == "__main__":
    main()
