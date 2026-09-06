import os
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image


class AstroImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, classes, transform=None):
        self.samples = samples
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.transform = transform

    @staticmethod
    def load_samples(root_dir):
        root = Path(root_dir).expanduser().resolve()
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        samples = []
        classes = set()

        # Enforce strict hierarchy: ProcessedData -> Telescope -> Class -> Image
        # Example: ProcessedData / SDSS / lens / image.png
        
        for telescope_dir in root.iterdir():
            # Skip files at the root level
            if not telescope_dir.is_dir():
                continue
                
            for class_dir in telescope_dir.iterdir():
                # Skip files at the telescope level (this prevents SDSS/Subaru from becoming classes)
                if not class_dir.is_dir():
                    continue
                    
                # The folder name here is 'lens', 'ring', etc.
                class_name = class_dir.name
                
                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        samples.append((str(file_path), class_name))
                        classes.add(class_name)

        classes = sorted(list(classes))
        return samples, classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[class_name]


class DataFactory:
    @staticmethod
    def create_loaders(cfg):
        d = cfg.get("data", {})
        root = d.get("root_dir", "./ProcessedData")
        img_size = d.get("image_size", 224)
        batch_size = d.get("batch_size", 32)
        val_split = float(d.get("val_split", 0.2))
        num_workers = int(d.get("num_workers", 4))
        augment = d.get("augment", True)

        # Base transforms applied to everything
        train_transforms = [transforms.Resize((img_size, img_size))]
        
        # Spatial and Color Augmentations (Ideal for Astro Data)
        if augment:
            train_transforms += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180), # Space has no "up", free rotation is great here
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            ]
            
        train_transforms += [
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        val_transforms = [
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        train_tf = transforms.Compose(train_transforms)
        val_tf = transforms.Compose(val_transforms)

        samples, classes = AstroImageDataset.load_samples(root)
        if not samples:
            raise RuntimeError(f"No images found in {root}. Check the data path and directory structure.")

        total = len(samples)
        val_count = int(total * val_split)
        train_count = total - val_count

        indices = torch.randperm(total).tolist()
        train_indices = indices[:train_count]
        val_indices = indices[train_count:]

        print(f"Found {total} images across {len(classes)} classes: {classes}")

        train_dataset = AstroImageDataset(samples, classes, transform=train_tf)
        val_dataset = AstroImageDataset(samples, classes, transform=val_tf)

        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        num_classes = len(classes)
        return train_loader, val_loader, num_classes, classes


if __name__ == '__main__':
    # simple smoke test when run directly
    cfg_path = Path("./runs/resnet_18.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
    else:
        cfg = {"data": {"root_dir": "./ProcessedData/"}}
    
    _, _, num_classes, classes = DataFactory.create_loaders(cfg)
    print(f"Successfully loaded {num_classes} classes.")