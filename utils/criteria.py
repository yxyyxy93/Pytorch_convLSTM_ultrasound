import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


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

        # Check the number of dimensions and average accordingly
        if len(average_ssim.shape) > 1:
            average_ssim = average_ssim.mean(dim=tuple(range(1, len(average_ssim.shape))))
        return average_ssim
