# Copyright 2023
# Xiaoyu Leo Yang
# NTU
# ==============================================================================

import numpy as np

__all__ = [
    "normalize",
    "resample_3d_array_numpy"
]


def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


def resample_3d_array_numpy(image_origin, image_noisy, new_shape):
    """
    Resample a 3D array to a new shape using simple nearest neighbor interpolation.
    Only resamples the height and width dimensions, keeping the depth unchanged.

    Args:
        data (numpy.ndarray):
        new_shape (tuple):

    Returns:
        numpy.ndarray: The resampled 3D array with shape
    """
    new_H, new_W, new_D = new_shape

    H_orig, W_orig, D = image_origin.shape
    # Compute the ratio for the height and width dimensions
    ratio_h = H_orig / new_H
    ratio_w = W_orig / new_H
    # Create a new array with the desired shape
    resampled_image_origin = np.zeros([D, new_H, new_W])
    for d in range(D):
        for h in range(new_H):
            for w in range(new_W):
                # Find the nearest neighbor indices
                orig_h = min(int(h * ratio_h), H_orig - 1)
                orig_w = min(int(w * ratio_w), W_orig - 1)
                # Assign the value from the nearest neighbor
                resampled_image_origin[d, h, w] = image_origin[orig_w, orig_h, d]

    H_orig, W_orig, D = image_noisy.shape
    # Compute the ratio for the height and width dimensions
    ratio_h = H_orig / new_H
    ratio_w = W_orig / new_H
    # Create a new array with the desired shape
    resampled_image_noisy = np.zeros([D, new_H, new_W])
    for d in range(D):
        for h in range(new_H):
            for w in range(new_W):
                # Find the nearest neighbor indices
                orig_h = min(int(h * ratio_h), H_orig - 1)
                orig_w = min(int(w * ratio_w), W_orig - 1)
                # Assign the value from the nearest neighbor
                resampled_image_noisy[d, h, w] = image_noisy[orig_h, orig_w, d]

    return resampled_image_origin, resampled_image_noisy