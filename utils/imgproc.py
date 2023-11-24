# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import random
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

__all__ = [
    "normalize_and_add_channel"
]


def normalize_and_add_channel(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.transpose(2, 0, 1)  # Reorder to (T, H, W)
    image = np.expand_dims(image, axis=1)  # Add channel dimension: (T, C=1, H, W)
    return image