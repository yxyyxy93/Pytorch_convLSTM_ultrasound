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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import json
import datetime
import atexit
import signal
import sys

from dataset import CUDAPrefetcher, CPUPrefetcher, TrainValidImageDataset, show_dataset_info
from utils_func.utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter
from utils_func import criteria

# Set mode for training
os.environ['MODE'] = 'train'
import config
import unet_model


def main():
    # Load datasets for each fold
    dataloaders_per_fold = load_dataset(num_folds=5)
    # Initialize the number of training epochs
    start_epoch = 0
    print("Load all datasets successfully.")
    # show_dataset_info(train_prefetcher, show_sample_slices=True)
    model, ema_model = build_model()
    print(f"Build `{config.d_arch_name}` model successfully.")

    # get the loss function class based on the string name
    criterion = getattr(criteria, config.loss_function)()
    criterion = criterion.to(device=config.device)
    val_crite = getattr(criteria, config.val_function)()
    val_crite = val_crite.to(device=config.device)
    print("Define all loss functions successfully.")
    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")
    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")
    print("Check whether to load pretrained model weights...")
    if config.pretrained_d_model_weights_path:
        model = load_state_dict(model, config.pretrained_d_model_weights_path)
        print(f"Loaded `{config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume_d_model_weights_path:
        model, ema_model, start_epoch, optimizer, scheduler = load_state_dict(
            model,
            config.pretrained_d_model_weights_path,
            ema_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")
    # Iterate over each fold
    for fold, (train_prefetcher, val_prefetcher) in enumerate(dataloaders_per_fold):
        print(f"Starting training on fold {fold + 1}")
        # Printing the size of train and validation data
        train_data_size = len(train_prefetcher)
        val_data_size = len(val_prefetcher)
        print(f"Size of Training Data: {train_data_size}, Size of Validation Data: {val_data_size}")

        # Get current date
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        # Create a experiment results
        results_dir = os.path.join("results", f"{config.exp_name}_{current_date}", f"_fold {fold + 1}")
        make_directory(results_dir)

        # Create training process log file
        writer = SummaryWriter(os.path.join("logs", config.exp_name))

        # Initialize the gradient scaler
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # Initialize lists to store metrics for each epoch
        lowest_val_loss = 1
        best_score = 0
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_scores = []
        epoch_val_scores = []

        for epoch in range(start_epoch, config.epochs):
            avg_train_loss, avg_train_score = train(model,
                                                    ema_model,
                                                    train_prefetcher,
                                                    criterion,
                                                    optimizer,
                                                    epoch,
                                                    scaler,
                                                    writer,
                                                    val_crite)  # Pass the SSIM model to train
            avg_val_loss, avg_val_score = validate(model,
                                                   val_prefetcher,
                                                   epoch,
                                                   writer,
                                                   criterion,  # Pass the loss criterion to validate
                                                   val_crite,
                                                   "Val")

            # After train and validate calls
            # Save the training and validation metrics
            epoch_train_losses.append(avg_train_loss)
            epoch_train_scores.append(avg_train_score)
            epoch_val_losses.append(avg_val_loss)
            epoch_val_scores.append(avg_val_score)

            metrics = {
                "train_losses": epoch_train_losses,
                "train_scores": epoch_train_scores,
                "val_losses": epoch_val_losses,
                "val_scores": epoch_val_scores
            }
            # Save to a JSON file
            results_file = os.path.join(results_dir, f'training_metrics.json')
            with open(results_file, 'w') as f:
                json.dump(metrics, f)
            print("\n")

            # Update LR
            scheduler.step()

             # Automatically save the model with the lowest validation loss
            is_best = avg_val_loss < lowest_val_loss
            is_last = (epoch + 1) == config.epochs
            if is_best:
                lowest_val_loss = min(avg_val_loss, lowest_val_loss)
                best_score = avg_val_score

            save_checkpoint({"epoch": epoch + 1,
                             "best_score": best_score,
                             "best_loss": lowest_val_loss,
                             "state_dict": model.state_dict(),
                             "ema_state_dict": ema_model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict()},
                            results_dir=results_dir,
                            best_file_name="d_best.pth.tar",
                            last_file_name="d_last.pth.tar",
                            is_best=is_best,
                            is_last=is_last
                            )

        print(f"Completed training on fold {fold + 1}")


def load_dataset(num_folds=5) -> list:
    # Load the full dataset
    full_dataset = TrainValidImageDataset(image_dir=config.image_dir,
                                          label_dir=config.label_dir,
                                          mode=config.mode)

    dataset_size = len(full_dataset)
    indices = torch.randperm(dataset_size).tolist()

    # Calculate the size of each fold
    fold_size = dataset_size // num_folds

    dataloaders_per_fold = []

    for fold in range(num_folds):
        # Determine the indices for this fold
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]
        val_indices = indices[fold * fold_size:(fold + 1) * fold_size]

        # Create subsets for training and validation
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Create dataloaders for each subset
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=True, drop_last=True,
                                  persistent_workers=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
                                pin_memory=True, drop_last=True, persistent_workers=True)
        # Now you can check if 'device' is set to "cpu"
        if config.device == torch.device("cpu"):
            train_prefetcher = CPUPrefetcher(train_loader)
            val_prefetcher = CPUPrefetcher(val_loader)
        else:
            train_prefetcher = CUDAPrefetcher(train_loader, config.device)
            val_prefetcher = CUDAPrefetcher(val_loader, config.device)
        dataloaders_per_fold.append((train_prefetcher, val_prefetcher))

    return dataloaders_per_fold


def build_model() -> [nn.Module, nn.Module]:
    model = unet_model.__dict__[config.d_arch_name](n_channels=config.input_dim,
                                                    n_classes=config.output_dim)

    model = model.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_model = AveragedModel(model, avg_fn=ema_avg)

    return model, ema_model


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
        ema_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        val_crite: any  # Add the SSIM computation function
) -> (float, float):  # Change return type to include both loss and SSIM
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    scores = AverageMeter("Score", ":6.6f")  # New meter for SSIM
    progress = ProgressMeter(batches, [batch_time, data_time, losses, scores], prefix=f"Epoch: [{epoch + 1}]")

    train_model.train()
    train_prefetcher.reset()
    end = time.time()

    for batch_index, batch_data in enumerate(train_prefetcher):
        data_time.update(time.time() - end)
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)
        train_model.zero_grad(set_to_none=True)

        with amp.autocast():
            output = train_model(lr)
            # output = output[:, 0]  # only x-y location : [B, 1, w, h]
            # gt = gt[:, 0]  # only x-y location : [B, 1, w, h]
            gt = gt.long()  # Ensure ground truth is of type long
            loss = criterion(output[:, 0], gt[:, 0], output[:, 1], gt[:, 1])
            score = val_crite(output[:, 0], gt[:, 0])  # Compute

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(train_model)

        losses.update(loss.item(), lr.size(0))
        scores.update(score.item(), lr.size(0))  # Update  meter

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config.train_print_frequency == 0:
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/Score", score.item(), batch_index + epoch * batches + 1)  # Log SSIM
            progress.display(batch_index + 1)

    avg_loss = losses.avg
    avg_ssim = scores.avg  # Calculate average SSIM
    return avg_loss, avg_ssim  # Return both average loss and SSIM


def validate(
        validate_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        criterion: nn.MSELoss,  # Add criterion for loss computation
        val_crite: any,
        mode: str
) -> (float, float):  # Change return type to include both loss and SSIM
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")  # New meter for loss
    scores = AverageMeter("Score", ":6.6f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, losses, scores], prefix=f"{mode}: ")

    validate_model.eval()
    data_prefetcher.reset()
    end = time.time()

    with torch.no_grad():
        for batch_index, batch_data in enumerate(data_prefetcher):
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            with amp.autocast():
                output = validate_model(lr)
                output = output[:, 0]  # only x-y location : [B, 1, w, h]
                gt = gt[:, 0]  # only x-y location : [B, 1, w, h]
                gt = gt.long()  # Ensure ground truth is of type long
                loss = criterion(output[:, 0], gt[:, 0], output[:, 1], gt[:, 1])
                score = val_crite(output[:, 0], gt[:, 0])  # Compute

            losses.update(loss.item(), lr.size(0))  # Update loss meter
            scores.update(score.item(), lr.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % config.valid_print_frequency == 0:
                writer.add_scalar(f"{mode}/Loss", loss.item(), epoch + 1)  # Log loss
                writer.add_scalar(f"{mode}/SSIM", score.item(), epoch + 1)
                progress.display(batch_index + 1)

    progress.display_summary()
    avg_loss = losses.avg
    avg_score = scores.avg  # Calculate average SSIM
    return avg_loss, avg_score  # Return both average loss and SSIM


# Function to release GPU resources
def cleanup():
    torch.cuda.empty_cache()


def signal_handler(sig, frame):
    print('You pressed Ctrl+C or terminated the script!')
    # Perform any necessary cleanup here
    sys.exit(0)


if __name__ == "__main__":
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Your main program logic starts here
    print("Program is running... Press Ctrl+C to terminate.")
    main()
    # Register the cleanup function to be called at program exit
    atexit.register(cleanup)
