# Copyright 2023
# Xiaoyu Leo Yang
# NTU
# ==============================================================================

import numpy as np
import random

__all__ = [
    "normalize",
    "resample_3d_array_numpy"
]


def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


def resample_3d_array_numpy(image_origin, image_noisy, new_shape, section_shape):
    """
       Resample a 3D array to a new shape using simple nearest neighbor interpolation and then
       randomly select a section of the new shape.

       Args:
           image_origin (numpy.ndarray): The original image array.
           image_noisy (numpy.ndarray): The noisy image array.
           new_shape (tuple): The new shape for resampling.
           section_shape (tuple): The shape of the section to be randomly selected.

       Returns:
           numpy.ndarray: The resampled and sectioned 3D arrays.
       """
    new_H, new_W, new_D = new_shape
    section_H, section_W, section_D = section_shape

    # Resample
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

    # Randomly select a starting point for the section
    start_h = random.randint(0, new_H - section_H)
    start_w = random.randint(0, new_W - section_W)
    start_d = random.randint(0, new_D - section_D)

    # Extract the section from the resampled images
    section_image_origin = resampled_image_origin[
                           start_d:start_d + section_D,
                           start_h:start_h + section_H,
                           start_w:start_w + section_W]
    section_image_noisy = resampled_image_noisy[
                          start_d:start_d + section_D,
                          start_h:start_h + section_H,
                          start_w:start_w + section_W]

    return section_image_origin, section_image_noisy
