import torch
import torch.nn as nn
import os
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) / half_dim
        freqs = torch.exp(-math.log(self.max_freq) * exponent)
        args = t[:, None].float() * freqs[None, :]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 != 0:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_images(images, path, nrow=8):
    from torchvision.utils import save_image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, path, nrow=nrow)