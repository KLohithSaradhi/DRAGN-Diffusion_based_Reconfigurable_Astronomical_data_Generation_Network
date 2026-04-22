import torch
import torch.nn as nn
from utils import SinusoidalTimeEmbedding

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        
        # Time injection mimicking your AstroUNet ResidualBlock
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
        # Kept the 1x1 conv exactly as requested
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, t_emb):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Process and inject time embedding into the spatial grid
        t_add = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_add
        
        # Flatten spatial dimensions into a sequence for Attention: (B, Seq, C)
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        attn_output, _ = self.attn(h, h, h)
        
        # Reshape sequence back to spatial grid: (B, C, H, W)
        attn_output = attn_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Apply the final 1x1 conv and residual connection
        return x + self.proj_out(attn_output)


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Unpack dynamic config parameters (No hardcoding)
        in_channels = config["in_channels"]
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        depth = config["depth"]
        self.latent_size = config.get("latent_size", 32) # Extracted but architecture naturally adapts
        
        time_emb_dim = hidden_dim * 4
        
        # Global Time Embedding logic from AstroUNet
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 1x1 Conv to map latent channels to the hidden dimension
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 1)
        
        # Pure stack of AttentionBlocks (Depth directly controls the number of blocks)
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, time_emb_dim) 
            for _ in range(depth)
        ])
        
        # Final GroupNorm and 1x1 Conv to map back to latent channels
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, in_channels, 1)
        )
        
        # CRITICAL FIX: Zero-initialize the final layer to prevent gradient explosion
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    def forward(self, x, t):
        # 1. Process Time
        t_emb = self.time_mlp(t)
        
        # 2. Project Input
        h = self.conv_in(x)
        
        # 3. Variable Attention Stack
        for block in self.blocks:
            h = block(h, t_emb)
            
        # 4. Project Output
        return self.conv_out(h)