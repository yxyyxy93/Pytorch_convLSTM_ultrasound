# Copyright 2023
# Xiaoyu Leo Yang
# NTU
# ==============================================================================

import numpy as np

__all__ = [
    "normalize_and_add_channel"
]


def normalize_and_add_channel(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.transpose(2, 0, 1)  # Reorder to (T, H, W)
    image = np.expand_dims(image, axis=1)  # Add channel dimension: (T, C=1, H, W)
    return image

