import torch
import torch.nn as nn
from utils import SinusoidalTimeEmbedding

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, time_emb_dim):
        super().__init__()
        # Path 1: Attention (The Router)
        self.norm1 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        # Path 2: MLP (The Brain) - CRITICAL ADDITION
        self.norm2 = nn.GroupNorm(8, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1)
        )

    def forward(self, x, t_emb):
        # --- Path 1: Time-Conditioned Attention ---
        B, C, H, W = x.shape
        h = self.norm1(x)
        
        t_add = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_add
        
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        attn_output, _ = self.attn(h, h, h)
        attn_output = attn_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = x + self.proj_out(attn_output)
        
        # --- Path 2: Feed Forward Processing ---
        return x + self.mlp(self.norm2(x))


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        in_channels = config["in_channels"]
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        depth = config["depth"]
        self.latent_size = config.get("latent_size", 32) 
        
        time_emb_dim = hidden_dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 1)
        
        # The Spatial Map (CRITICAL for pure attention)
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, self.latent_size, self.latent_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, time_emb_dim) 
            for _ in range(depth)
        ])
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, in_channels, 1)
        )
        
        # Zero-initialize the final layer to prevent gradient explosion
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.conv_in(x)
        
        # Inject the map into the signal
        h = h + self.pos_embed
        
        for block in self.blocks:
            h = block(h, t_emb)
            
        return self.conv_out(h)