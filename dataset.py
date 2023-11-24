# Copyright
# Xiaoyu (Leo) Yang
# Nanyang Technological University
# 2023
# ==============================================================================
import os
import queue
import threading

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.Read_CSV import read_csv_to_3d_array
from utils import imgproc

__all__ = [
    "TrainValidImageDataset",
    "PrefetchGenerator",
    "PrefetchDataLoader",
    "CPUPrefetcher", "CUDAPrefetcher",
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
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


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
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
