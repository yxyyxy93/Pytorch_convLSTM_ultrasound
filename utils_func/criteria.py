import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

__all__ = [
    "SSIM3D",
    "DiceLoss",
    "MulticlassDiceLoss",
    "myCrossEntropyLoss",
    "PixelAccuracy",
    "CombinedLoss"
]


def _check_tensor_shape(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    """Check if the dimensions of the two tensors are the same"""
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied tensors have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def _ssim_torch_3d(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor, window_size: int,
                   gaussian_kernel_window: torch.Tensor) -> torch.Tensor:
    """PyTorch's implementation of SSIM for 3D image tensors."""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    # Adapting Gaussian kernel for 3D tensors
    gaussian_kernel_window = gaussian_kernel_window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    gaussian_kernel_window = gaussian_kernel_window.repeat(1, 1, 1, 1,
                                                           window_size)  # [1, 1, window_size, window_size, window_size]
    gaussian_kernel_window = gaussian_kernel_window.to(device=raw_tensor.device, dtype=raw_tensor.dtype)

    # Convert to 5D tensors by adding a channel dimension
    raw_tensor_5d = raw_tensor.unsqueeze(1)  # [batch_size, 1, height, width, length]
    dst_tensor_5d = dst_tensor.unsqueeze(1)  # [batch_size, 1, height, width, length]

    # Calculate mean, variance, and covariance
    mu1 = F.conv3d(raw_tensor_5d, gaussian_kernel_window, padding=window_size // 2, groups=1)
    mu2 = F.conv3d(dst_tensor_5d, gaussian_kernel_window, padding=window_size // 2, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv3d(raw_tensor_5d.pow(2), gaussian_kernel_window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(dst_tensor_5d.pow(2), gaussian_kernel_window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv3d(raw_tensor_5d * dst_tensor_5d, gaussian_kernel_window, padding=window_size // 2,
                       groups=1) - mu1_mu2

    # Calculate SSIM
    ssim_numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ssim_denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = ssim_numerator / ssim_denominator

    return ssim_map.mean(dim=(2, 3, 4))


def _ssim_single_torch_3d(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor, crop_border: int, window_size: int,
                          gaussian_kernel_window: np.ndarray) -> torch.Tensor:
    """Wrapper for the SSIM 3D calculation in PyTorch."""
    _check_tensor_shape(raw_tensor, dst_tensor)

    # Convert numpy kernel to torch tensor
    gaussian_kernel_window = torch.from_numpy(gaussian_kernel_window).float()

    # Crop border pixels for 4D tensors (batchsize, height, width, length)
    # Cropping is applied to width and length dimensions
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]

    return _ssim_torch_3d(raw_tensor, dst_tensor, window_size, gaussian_kernel_window)


class SSIM3D(nn.Module):
    def __init__(self, crop_border: int = 10, window_size: int = 5, gaussian_sigma: float = 1.5) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.window_size = window_size

        # Create a 3D Gaussian kernel
        gaussian_kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
        self.gaussian_kernel_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())
        self.gaussian_kernel_window = np.outer(self.gaussian_kernel_window, gaussian_kernel.reshape(-1)).reshape(
            window_size, window_size, window_size)

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, length = raw_tensor.size()
        ssim_scores = []

        # Iterate over the height dimension
        for h in range(height):
            current_raw = raw_tensor[:, h, :, :].unsqueeze(1)  # Select height slice and add channel dimension
            current_dst = dst_tensor[:, h, :, :].unsqueeze(1)  # Select height slice and add channel dimension

            # Calculate SSIM for the current slice
            ssim_score = _ssim_single_torch_3d(current_raw, current_dst, self.crop_border, self.window_size,
                                               self.gaussian_kernel_window)
            ssim_scores.append(ssim_score)

        # Average the SSIM scores over all slices in the height dimension
        average_ssim = torch.mean(torch.stack(ssim_scores), dim=0)

        # If the tensor has more than one dimension, reduce it to a single value
        if len(average_ssim.shape) > 1:
            average_ssim = average_ssim.mean()

        return average_ssim


# -------------- loss functions
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Apply sigmoid to the inputs
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_mse=0.5, threshold=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.weight_dice = weight_dice
        self.weight_mse = weight_mse
        self.threshold = threshold

    def forward(self, input1, target1, input2, target2):
        """
        Calculate the combined Dice and MSE loss.
        :param input1: The input tensor for DiceLoss
        :param target1: The target tensor for DiceLoss
        :param input2: The input tensor for MSELoss
        :param target2: The target tensor for MSELoss
        :return: Weighted combined loss
        """
        loss_dice = self.dice_loss(input1, target1)
        # Convert input1 to a binary mask
        mask = (input1 > self.threshold).float()
        # Use mask to mask input2
        masked_input2 = input2 * mask
        target2 = target2.float()
        loss_mse = self.mse_loss(masked_input2.squeeze(), target2.squeeze())
        return self.weight_dice * loss_dice + self.weight_mse * loss_mse


class IoU(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoU, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        # IoU calculation
        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, weight=None):
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, y_pred, y_true):
        # y_pred shape: [B, T, C, H, W]
        # y_true shape: [B, T, H, W]

        y_pred = torch.softmax(y_pred, dim=2)  # Apply softmax along the class dimension
        B, T, C, H, W = y_pred.shape

        # One-hot encode y_true
        y_true_one_hot = torch.zeros(B, T, C, H, W, device=y_pred.device)
        y_true_one_hot.scatter_(2, y_true.unsqueeze(2), 1)  # Now y_true_one_hot is [B, T, C, H, W]

        dice_loss = 0.0
        for c in range(C):
            y_true_c = y_true_one_hot[:, :, c, ...]
            y_pred_c = y_pred[:, :, c, ...]

            intersection = 2.0 * (y_pred_c * y_true_c).sum(dim=[0, 1, 2, 3])
            union = y_pred_c.sum(dim=[0, 1, 2, 3]) + y_true_c.sum(dim=[0, 1, 2, 3])

            dice_c = (intersection + self.smooth) / (union + self.smooth)

            if self.weight is not None:
                dice_c *= self.weight[c]
            dice_loss += dice_c

        # Average over the classes
        dice_loss /= C

        return 1 - dice_loss.mean()


class myCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(myCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 ignore_index=ignore_index,
                                                 reduction=reduction)

    def forward(self, input_tensor, target):
        # Permute the input tensor to shape [B, C, T, H, W]
        input_permuted = input_tensor.permute(0, 2, 1, 3, 4)

        return self.cross_entropy(input_permuted, target)


class PixelAccuracy(nn.Module):
    def __init__(self):
        super(PixelAccuracy, self).__init__()

    def forward(self, y_pred, y_true):
        """
            Calculate pixel accuracy for multi-class segmentation.
            :param y_pred: The prediction tensor of shape [B, H, W]
            :param y_true: The ground truth tensor of shape [B, H, W]
            :return: Pixel accuracy
            """
        # Get the predicted class for each pixel
        threshold = 0.5
        predicted = (y_pred > threshold).long()

        # Calculate accuracy for each class
        correct = (predicted == y_true).sum()
        total = y_true.numel()

        return correct.float() / total


# Example usage
if __name__ == "__main__":
    # Dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 16, 16)  # Model's raw output (logits)
    targets = torch.randint(0, 2, (batch_size, 1, 16, 16)).float()  # Binary targets

    # Create an instance of the DiceLoss
    dice_loss = DiceLoss()

    # Calculate loss
    loss = dice_loss(predictions, targets)
    print("Dice Loss:", loss.item())

    # Create an instance of PixelAccuracy
    pixel_acc = PixelAccuracy()
    # Calculate accuracy
    accuracy = pixel_acc(predictions, targets)
    print("Pixel Accuracy:", accuracy.item())

    # Dummy data for the example
    input1 = torch.randn(4, 1, 16, 16)  # Example input for DiceLoss
    target1 = torch.randint(0, 2, (4, 1, 16, 16)).float()  # Example target for DiceLoss

    input2 = torch.randn(4, 1, 16, 16)  # Example input for MSELoss
    target2 = torch.randn(4, 1, 16, 16)  # Example target for MSELoss

    # Initialize the combined loss with weights
    combined_loss_function = CombinedLoss(weight_dice=0.5, weight_mse=0.5)

    # Calculate combined loss
    combined_loss = combined_loss_function(input1, target1, input2, target2)
    print("Weighted Combined Loss:", combined_loss.item())
