import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
import pandas as pd
import numpy as np
import os
from PIL import Image
import transforms as T

class DRAGNDataset(Dataset):
    """Base class for all datasets ensuring standard loader methods."""
    def __init__(self, transform=None):
        self.transform = transform

    def get_full_dataloader(self, batch_size=64, shuffle=True, num_workers=2):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def _get_subset_loader(self, indices, batch_size, shuffle, num_workers):
        subset = Subset(self, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def get_condition_dataloader(self, condition, batch_size=64, shuffle=True, num_workers=2):
        raise NotImplementedError


class AstroDataset(DRAGNDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(transform)
        self.root_dir = root_dir
        self.file_list = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.file_list.append((os.path.join(cls_dir, f), cls))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, cls_name = self.file_list[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.class_to_idx[cls_name]

    def get_condition_dataloader(self, class_name, batch_size=64, shuffle=True, num_workers=2):
        indices = [i for i, (p, c) in enumerate(self.file_list) if c == class_name]
        return self._get_subset_loader(indices, batch_size, shuffle, num_workers)


class MNISTDataset(DRAGNDataset):
    def __init__(self, root='./data', train=True, transform=None):
        super().__init__(transform)
        self.dataset = datasets.MNIST(root=root, train=train, download=True)
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

    def get_condition_dataloader(self, digit, batch_size=64, shuffle=True, num_workers=2):
        # Filter by specific digit (0-9)
        indices = (self.targets == int(digit)).nonzero(as_tuple=True)[0]
        return self._get_subset_loader(indices, batch_size, shuffle, num_workers)


class AlphabetDataset(DRAGNDataset):
    def __init__(self, csv_path, transform=None, sample_size=None):
        super().__init__(transform)
        df = pd.read_csv(csv_path)
        if sample_size:
            df = df.sample(sample_size)
        self.data = df
        self.labels = self.data.iloc[:, 0].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row[0])
        # Convert flat row (784 pixels) to 28x28 image
        img = row[1:].values.astype(np.uint8).reshape(28, 28)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_condition_dataloader(self, char_idx, batch_size=64, shuffle=True, num_workers=2):
        # char_idx: 0='A', 1='B', ..., 25='Z'
        indices = np.where(self.labels == int(char_idx))[0]
        return self._get_subset_loader(indices, batch_size, shuffle, num_workers)


class DataFactory:
    """Helper to build data loaders dynamically from config."""
    @staticmethod
    def create_loader(config):
        d_cfg = config["data"]
        dataset_type = d_cfg.get("dataset_type", "astro").lower()
        batch_size = d_cfg.get("batch_size", 32)
        condition = d_cfg.get("condition", None)
        root_dir = d_cfg.get("root_dir", "./data")

        # 1. Initialize Dataset & Transform based on type
        if dataset_type == "astro":
            transform = T.astro_hd
            dataset = AstroDataset(root_dir=root_dir, transform=transform)
            
        elif dataset_type == "mnist":
            transform = T.mnist_standard
            dataset = MNISTDataset(root=root_dir, train=True, transform=transform)
            
        elif dataset_type == "alphabet":
            transform = T.mnist_standard
            dataset = AlphabetDataset(csv_path=root_dir, transform=transform)
            
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # 2. Return the requested DataLoader (Conditional or Full)
        if condition is not None and str(condition).strip() != "":
            return dataset.get_condition_dataloader(condition, batch_size=batch_size)
            
        return dataset.get_full_dataloader(batch_size=batch_size)