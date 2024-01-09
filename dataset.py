# Copyright
# Xiaoyu (Leo) Yang
# Nanyang Technological University
# 2023
# ==============================================================================
import queue
import threading

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils_func import imgproc
from utils_func.Read_CSV import read_csv_to_3d_array

__all__ = [
    "TrainValidImageDataset",
    "PrefetchGenerator",
    "PrefetchDataLoader",
    "CPUPrefetcher", "CUDAPrefetcher",
    "show_dataset_info"
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        label_dir (str): Directory where the corresponding labels are stored.
        mode (str): Data set loading method, the training data set is for data enhancement, and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, label_dir: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        # Get all subdirectories in the image directory
        self.subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        # Create a mapping from dataset files to label files
        self.dataset_label_mapping = self._create_dataset_label_mapping()

    def _create_dataset_label_mapping(self):
        mapping = {}
        # Iterate through each subdirectory
        for subdir in self.subdirs:
            dataset_path = os.path.join(self.image_dir, subdir)
            label_path = os.path.join(self.label_dir, subdir)
            # Get all dataset and label files in the subdirectory
            dataset_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            label_file = [f for f in os.listdir(label_path) if f.startswith('structure')]
            # Map each dataset file to its corresponding label file
            for dataset_file in dataset_files:
                # Assuming the file names are the same except for the prefix 'structure'
                # and the extension '.00.csv'
                full_dataset_file = os.path.join(dataset_path, dataset_file)
                full_label_file = os.path.join(label_path, label_file[0])
                mapping[full_dataset_file] = full_label_file

        return mapping

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Use the mapping to get the dataset and label files
        dataset_file = list(self.dataset_label_mapping.keys())[batch_index]
        label_file = self.dataset_label_mapping[dataset_file]

        # Load the images
        image_noisy = read_csv_to_3d_array(dataset_file)
        image_origin = read_csv_to_3d_array(label_file)

        # print_statistics(image_origin, "Initial")
        # Assign 1 where the value is 7, and 0 otherwise in the 'image_origin' array
        # image_origin = np.where(image_origin == 7, 1, 0)
        # Step 1: Find the index of the first 7 in the 3rd dimension
        idx_of_7 = np.argmax(image_origin == 7, axis=2)
        # Initialize a mask with all zeros
        image_origin = np.zeros_like(image_origin, dtype=bool)
        # Step 2: Set the positions greater than the index to True
        for i in range(image_origin.shape[0]):
            for j in range(image_origin.shape[1]):
                image_origin[i, j, idx_of_7[i, j]:] = True

        # factor_ranges = ((0.8, 1.2),
        #                  (0.9, 1.1),
        #                  (0.9, 1.1))  # Ranges for the resize factors for each dimension
        # image_origin, image_noisy = imgproc.resize_and_restore_images(image_origin,
        #                                                               image_noisy,
        #                                                               factor_ranges=factor_ranges)

        # print_statistics(image_origin, "After Resize and Restore")
        new_shape = [21, 21, 256]  # smaller size to match both dataset: image_noisy
        section_shape = [16, 16, 256]  # random select a section
        image_origin, image_noisy = imgproc.resample_3d_array_numpy(image_origin,
                                                                    image_noisy,
                                                                    new_shape, section_shape)

        dilation_factors = (20, 1, 1)  # Example dilation factors
        image_origin = imgproc.dilate_3d_array(image_origin, dilation_factors)

        # print_statistics(image_origin, "After Rearranging 3D Array")
        # First Tensor: Location of Class 1 in terms of W and H
        location_matrix = np.any(image_origin == 1, axis=0)  # Shape: [W, H] for debug

        image_noisy = imgproc.normalize(image_noisy)
        image_noisy = image_noisy[np.newaxis, :, :, :]  # add a feature channel
        image_origin = image_origin[np.newaxis, :, :, :]  # add a feature channel

        # Convert location and depth matrices, and noisy image to PyTorch tensors
        location_tensor = torch.from_numpy(location_matrix).long()
        origin_tensor = torch.from_numpy(image_origin).float()
        noisy_tensor = torch.from_numpy(image_noisy).float()

        return {"gt": origin_tensor, "lr": noisy_tensor, "loc_xy": location_tensor}

    def __len__(self) -> int:
        return len(self.dataset_label_mapping)


class TestDataset(Dataset):
    """
    Define test dataset loading methods.
    Args:
        image_dir (str): Test dataset directory.
        label_dir (str): Directory where the corresponding labels are stored.
    """

    def __init__(self, image_dir: str, label_dir: str) -> None:
        super(TestDataset, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        # Get all subdirectories in the image directory
        self.subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        self.dataset_label_mapping = self._create_dataset_label_mapping()

    def _create_dataset_label_mapping(self):
        mapping = {}
        # Iterate through each subdirectory
        for subdir in self.subdirs:
            dataset_path = os.path.join(self.image_dir, subdir)
            label_path = os.path.join(self.label_dir, subdir)
            # Get all dataset and label files in the subdirectory
            dataset_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            label_files = [f for f in os.listdir(label_path) if f.startswith('structure')]
            # Map each dataset file to its corresponding label file
            for dataset_file in dataset_files:
                full_dataset_file = os.path.join(dataset_path, dataset_file)
                full_label_file = os.path.join(label_path, label_files[0])  # Adjust as needed
                mapping[full_dataset_file] = full_label_file
        return mapping

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor]:
        dataset_file = list(self.dataset_label_mapping.keys())[index]
        label_file = self.dataset_label_mapping[dataset_file]
        # Load the images
        image_noisy = read_csv_to_3d_array(dataset_file)
        image_origin = read_csv_to_3d_array(label_file)

        # Assign 1 where the value is 7, and 0 otherwise in the 'image_origin' array
        # image_origin = np.where(image_origin == 7, 1, 0)
        # Step 1: Find the index of the first 7 in the 3rd dimension
        idx_of_7 = np.argmax(image_origin == 7, axis=2)
        # Initialize a mask with all zeros
        image_origin = np.zeros_like(image_origin, dtype=bool)
        # Step 2: Set the positions greater than the index to True
        for i in range(image_origin.shape[0]):
            for j in range(image_origin.shape[1]):
                if idx_of_7[i, j] != 0:
                    image_origin[i, j, idx_of_7[i, j]:] = True

        # # print_statistics(image_origin, "After Condition Assignment")
        # factor_ranges = ((0.8, 1.2),
        #                  (0.9, 1.1),
        #                  (0.9, 1.1))  # Ranges for the resize factors for each dimension
        # image_origin, image_noisy = imgproc.resize_and_restore_images(image_origin,
        #                                                               image_noisy,
        #                                                               factor_ranges=factor_ranges)

        new_shape = [21, 21, 320]  # smaller size to match both dataset: image_noisy
        section_shape = [16, 16, 320]  # random select a section
        image_origin, image_noisy = imgproc.resample_3d_array_numpy(image_origin,
                                                                    image_noisy,
                                                                    new_shape, section_shape)

        # dilation_factors = (20, 1, 1)  # Example dilation factors
        # image_origin = imgproc.dilate_3d_array(image_origin, dilation_factors)

        # print_statistics(image_origin, "After Rearranging 3D Array")
        # First Tensor: Location of Class 1 in terms of W and H
        location_matrix = np.any(image_origin == 1, axis=0)  # Shape: [W, H] for debug

        image_noisy = imgproc.normalize(image_noisy)
        image_noisy = image_noisy[np.newaxis, :, :, :]  # add a feature channel
        image_origin = image_origin[np.newaxis, :, :, :]  # add a feature channel

        # Convert location and depth matrices, and noisy image to PyTorch tensors
        location_tensor = torch.from_numpy(location_matrix).long()
        origin_tensor = torch.from_numpy(image_origin).float()
        noisy_tensor = torch.from_numpy(image_noisy).float()

        return {"gt": origin_tensor, "lr": noisy_tensor, "loc_xy": location_tensor}

    def __len__(self) -> int:
        return len(self.dataset_label_mapping)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        # Using 'next' on the iterator, and catch the StopIteration exception
        try:
            data = next(self.dataloader_iter)
            return data
        except StopIteration:
            # Reinitialize the iterator and stop the iteration
            self.dataloader_iter = iter(self.dataloader)
            raise StopIteration

    def reset(self):
        self.dataloader_iter = iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            raise StopIteration

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


def show_dataset_info(data_loader, show_sample_slices=False):
    if not data_loader:
        print("DataLoader is empty.")
        return

    total_samples = 0
    for i, batch in enumerate(data_loader):
        # Access the 'gt' or 'lr' key
        data = batch['lr']
        total_samples += data.size(0)

        if i == 0 and show_sample_slices:
            print("Sample size:", data.size())
            sample = data[0]  # Get the first sample in the batch

            depth, height, width = sample.shape[0], sample.shape[2], sample.shape[3]

            # Extract middle slices
            xy_slice = sample[depth // 2, :, :, :].squeeze()
            yz_slice = sample[:, :, :, width // 2].squeeze()
            xz_slice = sample[:, :, height // 2, :].squeeze()

            fig, _ = plt.subplots(1, 3, figsize=(15, 4))  # Adjust for an additional subplot for the slider
            ax1 = plt.subplot(1, 3, 1)
            ax1.imshow(xy_slice.cpu().numpy(), cmap='gray')
            ax1.set_title('XY slice')
            plt.subplot(1, 3, 2)
            plt.imshow(yz_slice.cpu().numpy(), cmap='gray')
            plt.title('YZ slice')
            plt.subplot(1, 3, 3)
            plt.imshow(xz_slice.cpu().numpy(), cmap='gray')
            plt.title('XZ slice')

            plt.show()

    print("Total number of samples:", total_samples)


def print_statistics(image, label):
    print(f"Statistics for {label}:")
    print(f"Min: {image.min()}, Max: {image.max()}")
    print(f"Mean: {image.mean()}, Std: {image.std()}")
    unique, counts = np.unique(image, return_counts=True)
    print("Value distribution:", dict(zip(unique, counts)))
    print()


def find_block_center(array, value=1):
    """
    Find the center of a block with a specified value.
    """
    coords = np.argwhere(array == value)
    if coords.size == 0:
        raise ValueError(f"No element with value {value} found in the array.")
    center = coords.mean(axis=0).astype(int)
    return center


def plot_dual_orthoslices(data1, data2, value=1):
    """
    Plot orthoslices for two 3D arrays, centered on a block of a given value in the first array.
    """
    if data1.ndim != 3 or data2.ndim != 3:
        raise ValueError("Both data1 and data2 must be 3D arrays.")

    # Find the center of the block in the first array
    center = find_block_center(data1, value)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows for each 3D array
    # Plotting for the first array
    for i in range(3):
        ax[0, i].imshow(data1.take(indices=center[i], axis=i), cmap='gray', aspect='auto')
        ax[0, i].set_title(f'Array 1 - Slice {center[i]}')
        ax[0, i].axis('off')

    # Plotting for the second array
    for i in range(3):
        ax[1, i].imshow(data2.take(indices=center[i], axis=i), cmap='gray', aspect='auto')
        ax[1, i].set_title(f'Array 2 - Slice {center[i]}')
        ax[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import os

    # Set mode for testing
    os.environ['MODE'] = 'train'
    import config

    # from visualization import visualize_sample

    # ------------- visualize some samples
    # Prepare test dataset
    test_dataset = TestDataset(config.image_dir, config.label_dir)  # Adjust as per your dataset class
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False)  # Adjust batch_size and other parameters as needed

    for data in test_loader:
        input = data['lr'].to(config.device)
        gt = data['gt'].to(config.device)
        # Check if all elements in gt are zero
        if torch.all(gt.eq(0)):
            print("Skipping as gt is all zeros")
            continue
        print(input.shape)
        print(gt.shape)

        # Visualize the sample
        loc_xy = data['loc_xy'].to(config.device)
        if torch.all(loc_xy.eq(0)):
            print("Skipping as loc_xy is all zeros")
            continue

        plot_dual_orthoslices(gt.squeeze().numpy(), input.squeeze().numpy(), value=1)
