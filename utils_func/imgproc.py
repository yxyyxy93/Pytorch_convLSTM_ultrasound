# Copyright 2023
# Xiaoyu Leo Yang
# NTU
# ==============================================================================

import numpy as np
import scipy.ndimage
import random

__all__ = [
    "normalize",
    "resample_3d_array_numpy",
    "resize_and_restore_images"
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


def resize_and_restore_images(image1, image2, factor_ranges):
    """
    Apply randomly chosen stretch/shrink factors within specified ranges for each dimension
    to two 3D images and then resize them back to their respective original sizes.

    Args:
        image1 (numpy.ndarray): The first 3D image.
        image2 (numpy.ndarray): The second 3D image.
        factor_ranges (tuple of tuples): Three tuples defining the range of the resize factors
                                         for each dimension (height, width, depth).

    Returns:
        tuple: The processed 3D images.
    """
    # Choose a random factor for each dimension within the given ranges
    factors = tuple(random.uniform(*range) for range in factor_ranges)

    # Process both images with the same factors
    processed_image1 = resize_and_restore_single_image(image1, factors)
    processed_image2 = resize_and_restore_single_image(image2, factors)
    t = 0.5  # Replace with your threshold
    processed_image2 = (processed_image2 >= t).astype(np.int_)
    processed_image2 = (processed_image2 < t).astype(np.int_)

    return processed_image1, processed_image2


def resize_and_restore_single_image(image, factors, pad_mode='reflect'):
    orig_shape = image.shape
    new_shape = [max(1, int(s * factor)) for s, factor in zip(orig_shape, factors)]

    # Apply padding to the image before resizing to avoid edge artifacts
    padded_image = np.pad(image, pad_width=1, mode=pad_mode)
    new_padded_shape = [s + 2 for s in new_shape]  # Account for padding

    # Resize the padded image
    padded_resized_image = crop_or_pad_3d_center(padded_image, new_padded_shape)
    zoom_factors = [o / (n - 2) for o, n in zip(orig_shape, new_padded_shape)]  # Adjust zoom factors for padding
    padded_restored_image = scipy.ndimage.zoom(padded_resized_image, zoom=zoom_factors, order=3)

    # Crop the padding off the restored image to get back to the original size
    crop_slices = tuple(slice(1, -1) if z < 1 else slice(None) for z in zoom_factors)
    restored_image = padded_restored_image[crop_slices]

    return restored_image


def crop_or_pad_3d_center(image, target_shape):
    """
    Crop or pad the 3D image to a target shape, centered around the middle of the image.

    Args:
        image (numpy.ndarray): The 3D image to be cropped or padded.
        target_shape (tuple): The target shape (height, width, depth).

    Returns:
        numpy.ndarray: The cropped or padded 3D image.
    """
    # Ensure the image is 3D
    if len(image.shape) != 3:
        raise ValueError("Image must be 3D. Received shape: {}".format(image.shape))

    # Initialize output array
    output = np.zeros(target_shape)

    # Calculate cropping or padding needed
    crop_pad = [(t - o) for t, o in zip(target_shape, image.shape)]

    # Create index arrays for cropping or padding
    crop_slices = tuple(slice(max(0, -cp // 2), max(0, -cp // 2) + min(o, t)) for cp, o, t in zip(crop_pad, image.shape, target_shape))
    output_slices = tuple(slice(max(0, cp // 2), max(0, cp // 2) + min(o, t)) for cp, o, t in zip(crop_pad, image.shape, target_shape))

    # Perform cropping or padding
    output[np.ix_(*[np.arange(*s.indices(o)) for s, o in zip(output_slices, output.shape)])] = \
        image[np.ix_(*[np.arange(*s.indices(o)) for s, o in zip(crop_slices, image.shape)])]

    return output


if __name__ == "__main__":
    import dataset
    import os
    from torch.utils.data import DataLoader
    from visualization import visualize_sample

    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    # Navigate to the parent directory
    parent_dir = os.path.dirname(current_script_dir)
    # Construct the path to the 'sim_data' directory
    sim_data_dir = os.path.join(parent_dir, 'dataset', 'sim_data')
    sim_stru_dir = os.path.join(parent_dir, 'dataset', 'sim_struct')

    # Prepare test dataset
    test_dataset = dataset.TestDataset(sim_data_dir, sim_stru_dir)  # Adjust as per your dataset class
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False)  # Adjust batch_size and other parameters as needed

    for data in test_loader:
        inputs = data['lr']
        gt = data['gt']
        gt3D = data['gt3d']

    print(inputs.shape)
    print(gt3D.shape)

    datasets = [inputs.squeeze(), gt3D.squeeze()]

    factor_ranges = ((0.5, 1.5),
                     (0.5, 1.5),
                     (0.5, 1.5))  # Ranges for the resize factors for each dimension

    processed_image1, processed_image2 = resize_and_restore_images(inputs.squeeze(), gt3D.squeeze(), factor_ranges)

    print(processed_image1.shape)
    print(processed_image2.shape)

    visualize_sample(processed_image1.squeeze(), processed_image2.squeeze(), slice_idx=(84, 8, 8))
    visualize_sample(processed_image2.squeeze(), processed_image2.squeeze(), slice_idx=(128, 8, 8))
