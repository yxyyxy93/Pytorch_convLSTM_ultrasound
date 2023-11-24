# Copyright 2023
# Xiaoyu Leo Yang
# Nanyang Technological University
# ==============================================================================
import warnings

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import imgproc

__all__ = [
    "ssim",
    "SSIM3D",
]


# The following is the implementation of IQA method in Python, using CPU as processing device
def _check_image(raw_image: np.ndarray, dst_image: np.ndarray):
    """Check whether the size and type of the two images are the same

    Args:
        raw_image (np.ndarray): image data to be compared, BGR format, data range [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range [0, 255]

    """
    # check image scale
    assert raw_image.shape == dst_image.shape, \
        f"Supplied images have different sizes {str(raw_image.shape)} and {str(dst_image.shape)}"

    # check image type
    if raw_image.dtype != dst_image.dtype:
        warnings.warn(f"Supplied images have different dtypes{str(raw_image.shape)} and {str(dst_image.shape)}")


def _ssim_3d(raw_image: np.ndarray, dst_image: np.ndarray, window_size=11, window_sigma=1.5, C1=6.5025, C2=58.5225) -> float:
    """
    Compute the SSIM (Structural Similarity Index) for a single 3D volume.
    Args:
        raw_image: 3D numpy array.
        dst_image: 3D numpy array.
        window_size: Size of the gaussian kernel.
        window_sigma: Sigma of the gaussian kernel.
        C1: Variable to stabilize division with weak denominator.
        C2: Variable to stabilize division with weak denominator.
    Returns:
        SSIM index.
    """
    if not raw_image.shape == dst_image.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Create a 3D Gaussian kernel
    gauss_kernel_1d = cv2.getGaussianKernel(window_size, window_sigma).reshape(-1)
    window = np.outer(gauss_kernel_1d, gauss_kernel_1d)
    window = np.outer(window, gauss_kernel_1d).reshape(window_size, window_size, window_size)

    # Normalize the kernel
    window /= np.sum(window)

    # Convert images to double and apply window
    mu1 = cv2.filter2D(raw_image, -1, window)[window_size//2:-window_size//2+1, window_size//2:-window_size//2+1, window_size//2:-window_size//2+1]
    mu2 = cv2.filter2D(dst_image, -1, window)[window_size//2:-window_size//2+1, window_size//2:-window_size//2+1, window_size//2:-window_size//2+1]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(raw_image ** 2, -1, window)[window_size//2:-window_size//2+1, window_size//2:-window_size//2+1, window_size//2:-window_size//2+1] - mu1_sq
    sigma2_sq = cv2.filter2D(dst_image ** 2, -1, window)[window_size//2:-window_size//2+1, window_size//2:-window_size//2+1, window_size//2:-window_size//2+1] - mu2_sq
    sigma12 = cv2.filter2D(raw_image * dst_image, -1, window)[window_size//2:-window_size//2+1, window_size//2:-window_size//2+1, window_size//2:-window_size//2+1] - mu1_mu2

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(raw_image: np.ndarray, dst_image: np.ndarray, crop_border: int, only_test_y_channel: bool) -> np.ndarray:
    """Python implements the SSIM (Structural Similarity) function, which calculates single/multi-channel data

    Args:
        raw_image (np.ndarray): The image data to be compared, in BGR format, the data range is [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range is [0, 255]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        ssim_metrics (float): SSIM metrics for single channel

    """
    # Check if two images are similar in scale and type
    _check_image(raw_image, dst_image)

    # crop border pixels
    if crop_border > 0:
        raw_image = raw_image[crop_border:-crop_border, crop_border:-crop_border, ...]
        dst_image = dst_image[crop_border:-crop_border, crop_border:-crop_border, ...]

    # If you only test the Y channel, you need to extract the Y channel data of the YCbCr channel data separately
    if only_test_y_channel:
        raw_image = imgproc.expand_y(raw_image)
        dst_image = imgproc.expand_y(dst_image)

    # Convert data type to numpy.float64 bit
    raw_image = raw_image.astype(np.float64)
    dst_image = dst_image.astype(np.float64)

    channels_ssim_metrics = []
    for channel in range(raw_image.shape[3]):
        ssim_metrics = _ssim_3d(raw_image[..., channel], dst_image[..., channel])
        channels_ssim_metrics.append(ssim_metrics)
    ssim_metrics = np.mean(np.asarray(channels_ssim_metrics))

    return ssim_metrics


# The following is the IQA method implemented by PyTorch, using CUDA as the processing device
def _check_tensor_shape(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]

    """
    # Check if tensor scales are consistent
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def _ssim_torch_3d(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor, window_size: int, gaussian_kernel_window: np.ndarray) -> torch.Tensor:
    """
    PyTorch implementation of SSIM for 3D image tensors.
    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    # Adapt Gaussian kernel for 3D
    gaussian_kernel_window = torch.from_numpy(gaussian_kernel_window).unsqueeze(-1).repeat(1, 1, 1, window_size)
    gaussian_kernel_window = gaussian_kernel_window.expand(1, raw_tensor.size(1), window_size, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window.to(device=raw_tensor.device, dtype=raw_tensor.dtype)

    # Using F.conv3d for 3D tensors
    raw_mean = F.conv3d(raw_tensor.unsqueeze(0), gaussian_kernel_window, padding=window_size//2, groups=raw_tensor.size(0))
    dst_mean = F.conv3d(dst_tensor.unsqueeze(0), gaussian_kernel_window, padding=window_size//2, groups=dst_tensor.size(0))

    raw_mean_sq = raw_mean.pow(2)
    dst_mean_sq = dst_mean.pow(2)
    raw_dst_mean = raw_mean * dst_mean

    raw_variance = F.conv3d(raw_tensor.pow(2).unsqueeze(0), gaussian_kernel_window, padding=window_size//2, groups=raw_tensor.size(0)) - raw_mean_sq
    dst_variance = F.conv3d(dst_tensor.pow(2).unsqueeze(0), gaussian_kernel_window, padding=window_size//2, groups=dst_tensor.size(0)) - dst_mean_sq
    raw_dst_covariance = F.conv3d((raw_tensor * dst_tensor).unsqueeze(0), gaussian_kernel_window, padding=window_size//2, groups=raw_tensor.size(0)) - raw_dst_mean

    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_sq + dst_mean_sq + c1) * (raw_variance + dst_variance + c2)

    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = torch.mean(ssim_metrics, [2, 3, 4])

    return ssim_metrics.squeeze(0)



def _ssim_single_torch_3d(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor, crop_border: int, window_size: int, gaussian_kernel_window: np.ndarray) -> torch.Tensor:
    """
    Wrapper for the SSIM 3D calculation in PyTorch.
    """
    # Check if two tensor scales are similar
    _check_tensor_shape(raw_tensor, dst_tensor)

    # Crop border pixels for 3D tensors
    if crop_border > 0:
        raw_tensor = raw_tensor[:, crop_border:-crop_border, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, crop_border:-crop_border, crop_border:-crop_border, crop_border:-crop_border]

    # Convert data type to torch.float64 bit
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)

    ssim_metrics = _ssim_torch_3d(raw_tensor, dst_tensor, window_size, gaussian_kernel_window)

    return ssim_metrics


class SSIM3D(nn.Module):
    """
    PyTorch class for SSIM calculation on 3D image tensors.
    """
    def __init__(self, crop_border: int = 21, window_size: int = 11, gaussian_sigma: float = 1.5) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.window_size = window_size

        # Create a 3D Gaussian kernel
        gaussian_kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
        self.gaussian_kernel_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())
        self.gaussian_kernel_window = np.outer(self.gaussian_kernel_window, gaussian_kernel.reshape(-1)).reshape(window_size, window_size, window_size)

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> torch.Tensor:
        ssim_metrics = _ssim_single_torch_3d(raw_tensor, dst_tensor, self.crop_border, self.window_size, self.gaussian_kernel_window)
        return ssim_metrics
