# Copyright
# Xiaoyu (Leo) Yang
# Nanyang Technological University
# 2023
# ==============================================================================
import os
import queue
import threading
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from utils.Read_CSV import read_csv_to_3d_array
from utils import imgproc

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
            dataset_files = [f for f in os.listdir(dataset_path) if f.endswith('.00.csv')]
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

        # Normalize and add a channel dimension if necessary
        image_noisy = imgproc.normalize_and_add_channel(image_noisy)
        image_origin = imgproc.normalize_and_add_channel(image_origin)

        # Convert to PyTorch tensors
        noisy_tensor = torch.from_numpy(image_noisy).float()
        origin_tensor = torch.from_numpy(image_origin).float()

        origin_tensor = torch.squeeze(origin_tensor)

        return {"gt": origin_tensor, "lr": noisy_tensor}

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

        # Create a mapping from dataset files to label files
        self.dataset_label_mapping = self._create_dataset_label_mapping()

    def _create_dataset_label_mapping(self):
        mapping = {}
        dataset_files = [f for f in os.listdir(self.image_dir) if f.endswith('.csv')]
        label_files = [f for f in os.listdir(self.label_dir) if f.startswith('structure')]

        # Map each dataset file to its corresponding label file
        for dataset_file in dataset_files:
            full_dataset_file = os.path.join(self.image_dir, dataset_file)
            # Assuming corresponding label file has the same name in label directory
            full_label_file = os.path.join(self.label_dir, dataset_file.replace('.00.csv', '_structure.csv'))
            mapping[full_dataset_file] = full_label_file

        return mapping

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor]:
        dataset_file = list(self.dataset_label_mapping.keys())[index]
        label_file = self.dataset_label_mapping[dataset_file]

        # Load the images
        image_noisy = imgproc.read_csv_to_3d_array(dataset_file)
        image_origin = imgproc.read_csv_to_3d_array(label_file)

        # Normalize and add a channel dimension if necessary
        image_noisy = imgproc.normalize_and_add_channel(image_noisy)
        image_origin = imgproc.normalize_and_add_channel(image_origin)

        # Convert to PyTorch tensors
        noisy_tensor = torch.from_numpy(image_noisy).float()
        origin_tensor = torch.from_numpy(image_origin).float()

        origin_tensor = torch.squeeze(origin_tensor)

        return {"gt": origin_tensor, "lr": noisy_tensor}

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

            # Plotting
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(xy_slice.cpu().numpy(), cmap='gray')
            plt.title('XY slice')

            plt.subplot(1, 3, 2)
            plt.imshow(yz_slice.cpu().numpy(), cmap='gray')
            plt.title('YZ slice')

            plt.subplot(1, 3, 3)
            plt.imshow(xz_slice.cpu().numpy(), cmap='gray')
            plt.title('XZ slice')

            plt.show()

    print("Total number of samples:", total_samples)
