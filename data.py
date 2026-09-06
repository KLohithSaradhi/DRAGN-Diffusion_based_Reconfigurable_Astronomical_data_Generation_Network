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

        # --- 1. DYNAMIC TRANSFORM SELECTION ---
        # Default fallbacks so old YAMLs without the 'transform' key still work
        default_transforms = {
            "astro": "astro_hd",
            "astro raw": "astro_raw_hd",
            "mnist": "mnist_standard",
            "alphabet": "mnist_standard"
        }
        
        # Get the transform name from config, or use the default
        transform_name = d_cfg.get("transform", default_transforms.get(dataset_type))
        
        # Dynamically fetch the transform from the transforms.py module
        try:
            transform = getattr(T, transform_name)
        except AttributeError:
            raise ValueError(f"Transform '{transform_name}' was not found in transforms.py!")

        # --- 2. INITIALIZE DATASET ---
        if dataset_type == "astro":
            dataset = AstroRawDataset(root_dir=root_dir, transform=transform)

        elif dataset_type == "astro raw":
            # Assuming you want the same high-res transforms as the base Astro dataset
            transform = T.astro_raw_hd 
            dataset = AstroRawDataset(root_dir=root_dir, transform=transform)
            
        elif dataset_type == "mnist":
            dataset = MNISTDataset(root=root_dir, train=True, transform=transform)
            
        elif dataset_type == "alphabet":
            dataset = AlphabetDataset(csv_path=root_dir, transform=transform)
            
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # --- 3. RETURN DATALOADER ---
        if condition is not None and str(condition).strip() != "":
            return dataset.get_condition_dataloader(condition, batch_size=batch_size)
            
        return dataset.get_full_dataloader(batch_size=batch_size)

class AstroRawDataset(DRAGNDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(transform)
        self.root_dir = root_dir
        self.file_list = []
        
        # 1. Discover all instruments dynamically (ignoring hidden Unix files)
        self.instruments = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
        ])
        self.instrument_to_idx = {inst: i for i, inst in enumerate(self.instruments)}
        
        # 2. Discover all classes globally (across all instruments)
        class_set = set()
        for inst in self.instruments:
            inst_dir = os.path.join(root_dir, inst)
            classes = [
                d for d in os.listdir(inst_dir) 
                if os.path.isdir(os.path.join(inst_dir, d)) and not d.startswith('.')
            ]
            class_set.update(classes)
            
            # 3. Build the file manifest
            for cls in classes:
                cls_dir = os.path.join(inst_dir, cls)
                for f in os.listdir(cls_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.file_list.append((os.path.join(cls_dir, f), inst, cls))
                        
        self.classes = sorted(list(class_set))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, inst_name, cls_name = self.file_list[idx]
        
        # FIX: Force 1-channel Grayscale ("L") to preserve instrument physics
        # No try/except safety nets here. If a file is corrupted, it crashes loudly.
        img = Image.open(path)
        
        if self.transform: 
            img = self.transform(img)
            
        # Return a dictionary of labels so the training loop can access both
        labels = {
            "instrument": self.instrument_to_idx[inst_name],
            "class": self.class_to_idx[cls_name]
        }
        return img, labels

    def get_condition_dataloader(self, condition, batch_size=64, shuffle=True, num_workers=2):
        # Flexible filtering: Allows conditioning by either Instrument OR Class
        inst, cls = condition.split(':')
        if inst and cls:
            indices = [
                i for i, (p, inst_name, cls_name) in enumerate(self.file_list) 
                if inst == inst_name and cls == cls_name
            ]
        elif inst:
            indices = [
                i for i, (p, inst_name, cls_name) in enumerate(self.file_list) 
                if inst == inst_name
            ]
        elif cls:
            indices = [
                i for i, (p, inst_name, cls_name) in enumerate(self.file_list) 
                if cls == cls_name
            ]
        else:
            raise ValueError("Condition must specify at least an instrument or a class (e.g., 'HST:Galaxy' or 'HST:' or ':Galaxy').")
        # indices = [
        #     i for i, (p, inst, cls) in enumerate(self.file_list) 
        #     if condition == inst or condition == cls
        # ]
        
        if not indices:
            raise ValueError(f"Condition '{condition}' found no matching instruments or classes in AstroRawDataset.")
            
        return self._get_subset_loader(indices, batch_size, shuffle, num_workers)