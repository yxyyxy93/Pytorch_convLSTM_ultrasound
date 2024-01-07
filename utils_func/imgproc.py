# Copyright 2023
# Xiaoyu Leo Yang
# NTU
# ==============================================================================

import numpy as np
import scipy.ndimage
import random
from scipy.ndimage import binary_dilation

__all__ = [
    "normalize",
    "resample_3d_array_numpy",
    "resize_and_restore_images"
]


def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


def resample_3d_array_numpy(image_origin, image_noisy, new_shape, section_shape):
    new_H, new_W, new_D = new_shape
    section_H, section_W, section_D = section_shape

    # Resample the original image
    H_orig, W_orig, D_orig = image_origin.shape
    ratio_h = (H_orig - 1) / (new_H - 1)
    ratio_w = (W_orig - 1) / (new_W - 1)
    ratio_d = (D_orig - 1) / (new_D - 1)
    resampled_image_origin = np.zeros([new_D, new_H, new_W])
    for d in range(new_D):
        for h in range(new_H):
            for w in range(new_W):
                orig_h = min(int(round(h * ratio_h)), H_orig - 1)
                orig_w = min(int(round(w * ratio_w)), W_orig - 1)
                orig_d = min(int(round(d * ratio_d)), D_orig - 1)
                resampled_image_origin[d, h, w] = image_origin[orig_h, orig_w, orig_d]

    # Resample the noisy image
    H_orig, W_orig, D_orig = image_noisy.shape
    ratio_h = (H_orig - 1) / (new_H - 1)
    ratio_w = (W_orig - 1) / (new_W - 1)
    ratio_d = (D_orig - 1) / (new_D - 1)
    resampled_image_noisy = np.zeros([new_D, new_H, new_W])
    for d in range(new_D):
        for h in range(new_H):
            for w in range(new_W):
                orig_h = min(int(round(h * ratio_h)), H_orig - 1)
                orig_w = min(int(round(w * ratio_w)), W_orig - 1)
                orig_d = min(int(round(d * ratio_d)), D_orig - 1)
                resampled_image_noisy[d, w, h] = image_noisy[orig_h, orig_w, orig_d]

    # Select a random section from the resampled image
    start_h = np.random.randint(0, new_H - section_H + 1)
    start_w = np.random.randint(0, new_W - section_W + 1)
    start_d = np.random.randint(0, new_D - section_D + 1)

    section_origin = resampled_image_origin[
                     start_d:start_d + section_D,
                     start_h:start_h + section_H,
                     start_w:start_w + section_W]
    section_noisy = resampled_image_noisy[
                    start_d:start_d + section_D,
                    start_h:start_h + section_H,
                    start_w:start_w + section_W]

    return section_origin, section_noisy


def rearrange_3d_array_numpy(image_origin, image_noisy):
    """
       Rearrange 3D array
       Args:
           image_origin (numpy.ndarray): The original image array.
           image_noisy (numpy.ndarray): The noisy image array.
       Returns:
           numpy.ndarray: The resampled and sectioned 3D arrays.
       """
    # Resample
    H_orig, W_orig, D = image_origin.shape
    # Create a new array with the desired shape
    resampled_image_origin = np.zeros([D, H_orig, W_orig])
    for d in range(D):
        for h in range(H_orig):
            for w in range(W_orig):
                # Assign the value from the nearest neighbor
                resampled_image_origin[d, h, w] = image_origin[w, h, d]

    H_orig, W_orig, D = image_noisy.shape
    resampled_image_noisy = np.zeros([D, H_orig, W_orig])
    for d in range(D):
        for h in range(H_orig):
            for w in range(W_orig):
                # Assign the value from the nearest neighbor
                resampled_image_noisy[d, h, w] = image_noisy[h, w, d]

    return resampled_image_origin, resampled_image_noisy


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
    processed_image1 = (processed_image1 >= t).astype(np.int_)
    processed_image1 = (processed_image1 < t).astype(np.int_)

    return processed_image1, processed_image2


def resize_and_restore_single_image(image, factors, pad_mode='reflect'):
    orig_shape = image.shape

    # Calculate new shape after scaling
    new_shape = [max(1, int(s * factor)) for s, factor in zip(orig_shape, factors)]

    # Apply padding to the image before resizing to avoid edge artifacts
    padding = [(1, 1) for _ in range(len(orig_shape))]
    padded_image = np.pad(image, pad_width=padding, mode=pad_mode)

    # Resize the padded image
    zoom_factors = [ns / s for ns, s in zip(new_shape, orig_shape)]
    padded_resized_image = scipy.ndimage.zoom(padded_image, zoom=zoom_factors, order=3)

    # Crop or pad the resized image to match the original size
    final_image = crop_or_pad_3d_center(padded_resized_image, orig_shape)

    return final_image


def crop_or_pad_3d_center(image, target_shape):
    """
    Crop or pad the 3D image to a target shape, centered around the middle of the image.
    """
    # Ensure the image is 3D
    if len(image.shape) != 3:
        raise ValueError("Image must be 3D. Received shape: {}".format(image.shape))

    # Initialize output array
    output = np.zeros(target_shape)
    # Calculate cropping or padding needed
    crop_pad = [(t - o) for t, o in zip(target_shape, image.shape)]

    # Create index arrays for cropping or padding
    crop_slices = tuple(
        slice(max(0, -cp // 2), max(0, -cp // 2) + min(o, t)) for cp, o, t in zip(crop_pad, image.shape, target_shape))
    output_slices = tuple(
        slice(max(0, cp // 2), max(0, cp // 2) + min(o, t)) for cp, o, t in zip(crop_pad, image.shape, target_shape))

    # Perform cropping or padding
    output[np.ix_(*[np.arange(*s.indices(o)) for s, o in zip(output_slices, output.shape)])] = \
        image[np.ix_(*[np.arange(*s.indices(o)) for s, o in zip(crop_slices, image.shape)])]

    return output


def dilate_3d_array(array_3d, dilation_factors):
    """
    Dilate a 3D numpy array in three different directions with specified factors.

    Args:
    array_3d (numpy.ndarray): The input 3D array to be dilated.
    dilation_factors (tuple): A tuple of three integers representing the dilation factors for each direction (z, y, x).

    Returns:
    numpy.ndarray: The dilated 3D array with values as 0 and 1.
    """
    # Create a structuring element based on dilation factors
    structuring_element = np.ones(dilation_factors, dtype=bool)

    # Apply dilation
    dilated_array = binary_dilation(array_3d, structure=structuring_element)

    # Convert boolean array to integer array (True to 1, False to 0)
    dilated_array = dilated_array.astype(int)

    return dilated_array


if __name__ == "__main__":
    import dataset
    import os
    from torch.utils.data import DataLoader
    from visualization import visualize_sample

    # Example usage
    array_3d = np.random.randint(0, 2, (7, 7, 7))  # Example 3D array
    dilation_factors = (2, 3, 1)  # Example dilation factors
    dilated_array = dilate_3d_array(array_3d, dilation_factors)

    print("Original Array:\n", array_3d)
    print("Dilated Array - Original Array:\n", dilated_array - array_3d)

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
        gt = data['loc_xy']
        gt3D = data['gt']

    print(inputs.shape)
    print(gt3D.shape)

    datasets = [inputs.squeeze(), gt3D.squeeze()]

    factor_ranges = ((0.5, 1.5),
                     (0.5, 1.5),
                     (0.5, 1.5))  # Ranges for the resize factors for each dimension

    processed_image1, processed_image2 = resize_and_restore_images(inputs.squeeze(), gt3D.squeeze(), factor_ranges)

    print(processed_image1.shape)
    print(processed_image2.shape)
    #
    # visualize_sample(processed_image1.squeeze(), processed_image2.squeeze(), slice_idx=(84, 8, 8))
    # visualize_sample(processed_image2.squeeze(), processed_image2.squeeze(), slice_idx=(128, 8, 8))
