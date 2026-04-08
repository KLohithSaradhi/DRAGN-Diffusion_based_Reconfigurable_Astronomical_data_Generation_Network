import torch
import torch.nn as nn
import math
from utils import SinusoidalTimeEmbedding

class TransformerDiffusionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x, t_emb):
        x_time = x + t_emb.unsqueeze(1) 
        norm_x = self.norm1(x_time)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))

class DiffusionTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_channels, self.latent_size, self.patch_size = params["in_channels"], params["latent_size"], params.get("patch_size", 2)
        self.hidden_dim, self.num_heads, self.depth = params["hidden_dim"], params.get("num_heads", 8), params.get("depth", 6)
        
        self.num_patches = (self.latent_size // self.patch_size) ** 2
        patch_dim = self.in_channels * (self.patch_size ** 2)
        
        self.patch_embed = nn.Linear(patch_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.blocks = nn.ModuleList([TransformerDiffusionBlock(self.hidden_dim, self.num_heads) for _ in range(self.depth)])
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.decoder_pred = nn.Linear(self.hidden_dim, patch_dim)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, self.num_patches, -1)

    def depatchify(self, x):
        B, N, _ = x.shape
        p = self.patch_size
        h_w = int(math.sqrt(N))
        x = x.view(B, h_w, h_w, self.in_channels, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, self.in_channels, h_w * p, h_w * p)

    def forward(self, x, t):
        t_emb = self.time_mlp(t) 
        x = self.patch_embed(self.patchify(x)) + self.pos_embed
        for block in self.blocks: x = block(x, t_emb)
        return self.depatchify(self.decoder_pred(self.norm_out(x)))