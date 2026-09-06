import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class LoRAConv2dWrapper(nn.Module):
    def __init__(self, original_layer: nn.Conv2d, rank=4, alpha=1.0):
        super().__init__()
        self.scaling = alpha / rank
        self.original_layer = original_layer
        
        for p in self.original_layer.parameters():
            p.requires_grad = False
            
        self.lora_A = nn.Conv2d(
            original_layer.in_channels, rank, original_layer.kernel_size, 
            original_layer.stride, original_layer.padding, 
            dilation=original_layer.dilation, groups=1, bias=False
        )
        self.lora_B = nn.Conv2d(
            rank, original_layer.out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

class LoRATransposeConv2dWrapper(nn.Module):
    def __init__(self, original_layer: nn.ConvTranspose2d, rank=4, alpha=1.0):
        super().__init__()
        self.scaling = alpha / rank
        self.original_layer = original_layer
        
        for p in self.original_layer.parameters():
            p.requires_grad = False
            
        self.lora_A = nn.ConvTranspose2d(
            original_layer.in_channels, rank, original_layer.kernel_size, 
            original_layer.stride, original_layer.padding, 
            dilation=original_layer.dilation, groups=1, bias=False
        )
        self.lora_B = nn.ConvTranspose2d(
            rank, original_layer.out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

class LoRAAttentionWrapper(nn.Module):
    def __init__(self, original_attn: nn.MultiheadAttention, rank=4, alpha=1.0):
        super().__init__()
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        self.original_in_proj_weight = original_attn.in_proj_weight
        self.original_in_proj_bias = original_attn.in_proj_bias
        self.original_out_proj = original_attn.out_proj
        
        self.original_in_proj_weight.requires_grad = False
        if self.original_in_proj_bias is not None: 
            self.original_in_proj_bias.requires_grad = False
        for p in self.original_out_proj.parameters(): 
            p.requires_grad = False
            
        self.lora_q = LoRALinear(self.embed_dim, self.embed_dim, rank, alpha)
        self.lora_v = LoRALinear(self.embed_dim, self.embed_dim, rank, alpha)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        B, Seq, C = query.shape
        qkv = F.linear(query, self.original_in_proj_weight, self.original_in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q + self.lora_q(query)
        v = v + self.lora_v(query)
        
        q = q.reshape(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).reshape(B, Seq, C)
        return self.original_out_proj(attn_output), None

class LoRAManager:
    @staticmethod
    def inject_lora(model, rank=16, alpha=16.0):
        lora_params = []
        
        def _inject(module):
            for name, child in module.named_children():
                # 1. Inject into Standard Convolutions
                if isinstance(child, nn.Conv2d):
                    # Skip the absolute entrance/exit to avoid breaking the latent space mappings
                    if name in ['conv_in', 'conv_out']: 
                        continue
                    wrapper = LoRAConv2dWrapper(child, rank=rank, alpha=alpha)
                    setattr(module, name, wrapper)
                    lora_params.extend(list(wrapper.lora_A.parameters()))
                    lora_params.extend(list(wrapper.lora_B.parameters()))
                    
                # 2. Inject into Transpose Convolutions (Up-blocks for high-res details)
                elif isinstance(child, nn.ConvTranspose2d):
                    wrapper = LoRATransposeConv2dWrapper(child, rank=rank, alpha=alpha)
                    setattr(module, name, wrapper)
                    lora_params.extend(list(wrapper.lora_A.parameters()))
                    lora_params.extend(list(wrapper.lora_B.parameters()))
                    
                # 3. Inject into Attention (Bottleneck spatial routing)
                elif isinstance(child, nn.MultiheadAttention):
                    wrapper = LoRAAttentionWrapper(child, rank=rank, alpha=alpha)
                    setattr(module, name, wrapper)
                    lora_params.extend(list(wrapper.lora_q.parameters()))
                    lora_params.extend(list(wrapper.lora_v.parameters()))
                    
                # 4. Keep digging down the tree
                else: 
                    _inject(child)
                    
        _inject(model)
        return lora_params

    @staticmethod
    def save_weights(model, filepath):
        # Grabs all injected lora parameter dicts dynamically
        lora_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
        torch.save(lora_dict, filepath)

    @staticmethod
    def load_weights(model, filepath, device="cuda"):
        model.load_state_dict(torch.load(filepath, map_location=device), strict=False)