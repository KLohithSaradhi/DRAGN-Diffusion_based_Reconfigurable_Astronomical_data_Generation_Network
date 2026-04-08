import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class ConvolvePSF(nn.Module):
    def __init__(self, size=5, channels=3):
        super().__init__()
        # PyTorch expects weight shape: (groups, in_channels/groups, kH, kW)
        weight = torch.zeros((channels, 1, size, size))
        for i in range(size):
            weight[:, 0, i, i] = 1
            weight[:, 0, i, size-i-1] = 1
            
        # Normalize each channel's filter independently
        weight = weight / weight.sum(dim=(2, 3), keepdim=True)
        
        self.register_buffer('weight', weight)
        self.pad = size // 2
        self.channels = channels

    def __call__(self, x):
        x = x.unsqueeze(0) 
        # groups=self.channels applies the blur to R, G, and B independently
        x = F.conv2d(x, self.weight, padding=self.pad, groups=self.channels)
        noise = torch.randn_like(x) * 0.02
        return (x + noise).squeeze(0)

mnist_standard = transforms.Compose([
    transforms.ToTensor(), transforms.Pad(2), transforms.Normalize((0.5,), (0.5,)) 
])

astro_hd = transforms.Compose([
    transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

astro_distorted = transforms.Compose([
    transforms.Resize((512, 512)), transforms.ToTensor(), ConvolvePSF(size=5), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])