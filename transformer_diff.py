import torch
import torch.nn as nn
from utils import SinusoidalTimeEmbedding

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.residual_conv = nn.Identity() if in_channels == out_channels else \
                             nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        h = self.conv1(x)
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        h = self.conv2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        attn_output, _ = self.attn(h, h, h)
        attn_output = attn_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.proj_out(attn_output)


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- Parameter Parsing from your Config ---
        in_channels = config["in_channels"]
        base_channels = config["hidden_dim"]  # Maps config's hidden_dim to U-Net base_channels
        num_heads = config["num_heads"]
        depth = config["depth"]               # Controls the number of Attention Blocks
        ch_multis = config.get("ch_multis", [1, 2, 4]) # Defaults to standard multipliers if not explicitly in config
        
        time_emb_dim = base_channels * 4
        
        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # --- Down Path ---
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i, mult in enumerate(ch_multis):
            out_channels = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(current_channels, out_channels, time_emb_dim),
                ResidualBlock(out_channels, out_channels, time_emb_dim),
                nn.Conv2d(out_channels, out_channels, 2, stride=2, padding=0) if i != len(ch_multis) - 1 else nn.Identity()
            ]))
            current_channels = out_channels

        # --- Dynamic Bottleneck (Where depth applies) ---
        bottleneck_channels = current_channels
        self.bottleneck = nn.ModuleList()
        
        # 1. Enter the bottleneck with a ResBlock
        self.bottleneck.append(ResidualBlock(bottleneck_channels, bottleneck_channels, time_emb_dim))
        
        # 2. Add exactly 'depth' number of Attention Blocks based on config
        for _ in range(depth):
            self.bottleneck.append(AttentionBlock(bottleneck_channels, num_heads))
            
        # 3. Exit the bottleneck with a ResBlock
        self.bottleneck.append(ResidualBlock(bottleneck_channels, bottleneck_channels, time_emb_dim))
        
        # --- Up Path ---
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_multis)):
            out_channels = int(base_channels * mult / 2) 
            input_channels = out_channels * 4 
            out_channels = base_channels if i == len(ch_multis) - 1 else out_channels
            
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(input_channels, out_channels, time_emb_dim),
                ResidualBlock(out_channels , out_channels, time_emb_dim),
                nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2, padding=0) if i != len(ch_multis) - 1 else nn.Identity()
            ]))

        # --- Output Mapping ---
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.conv_in(x)
        
        skip_connections = []
        for res_block1, res_block2, downsample in self.down_blocks:
            x = res_block1(x, time_emb)
            x = res_block2(x, time_emb)
            skip_connections.append(x)
            x = downsample(x)
            
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else: 
                x = layer(x)

        for i, (res_block1, res_block2, upsample) in enumerate(self.up_blocks):
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = res_block1(x, time_emb)
            x = res_block2(x, time_emb)
            x = upsample(x)
            
        return self.conv_out(x)