# Copyright 2023
# Xiaoyu Leo Yang
# NTU
# ==============================================================================

import numpy as np

__all__ = [
    "normalize_and_add_channel",
    "resample_3d_array_numpy"
]


def normalize_and_add_channel(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.transpose(2, 0, 1)  # Reorder to (T, H, W)
    image = np.expand_dims(image, axis=1)  # Add channel dimension: (T, C=1, H, W)
    return image


def resample_3d_array_numpy(data, new_shape):
    """
    Resample a 3D array to a new shape using simple nearest neighbor interpolation.
    Only resamples the height and width dimensions, keeping the depth unchanged.

    Args:
        data (numpy.ndarray):
        new_shape (tuple):

    Returns:
        numpy.ndarray: The resampled 3D array with shape
    """
    new_H, new_W, _ = new_shape
    H_orig, W_orig, D = data.shape

    # Compute the ratio for the height and width dimensions
    ratio_h = H_orig / new_H
    ratio_w = W_orig / new_W

    # Create a new array with the desired shape
    resampled_data = np.zeros([D, new_H, new_W])

    for d in range(D):
        for h in range(new_H):
            for w in range(new_W):
                # Find the nearest neighbor indices
                orig_h = min(int(h * ratio_h), H_orig - 1)
                orig_w = min(int(w * ratio_w), W_orig - 1)

                # Assign the value from the nearest neighbor
                resampled_data[d, w, h] = data[orig_h, orig_w, d]

    return resampled_data



