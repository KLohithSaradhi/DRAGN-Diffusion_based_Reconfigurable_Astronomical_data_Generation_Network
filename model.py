import torch
import torch.nn as nn
from tqdm import tqdm
import torch
from scipy.optimize import linear_sum_assignment

class LatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, unet):
        super().__init__()
        self.ae = autoencoder
        self.unet = unet
        if self.ae is not None:
            for p in self.ae.parameters(): p.requires_grad = False
            self.ae.eval()

    @torch.no_grad()
    def encode(self, images): 
        if self.ae == None: return images
        return self.ae.encoder(images)

    @torch.no_grad()
    def decode(self, latents): 
        if self.ae == None: return latents
        return self.ae.decoder(latents)

    def forward(self, noisy_latents, t): return self.unet(noisy_latents, t)

    def compute_loss(self, images, scheduler, criterion):
        latents = self.encode(images)
        device = latents.device
        t = torch.randint(0, scheduler.timesteps, (latents.shape[0],), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.q_sample(latents, t, noise)
        noise_pred = self(noisy_latents, t)
        return criterion(noise_pred, noise)
    
    def compute_flow_loss(self, images, criterion):
        # 1. Get Data (x_1) - This uses the AE bypass we built perfectly!
        x_1 = self.encode(images)
        device = x_1.device
        batch_size = x_1.shape[0]
        
        # 2. Get Random Noise (x_0)
        x_0 = torch.randn_like(x_1)
        
        # --- 3. MINIBATCH OPTIMAL TRANSPORT ---
        # Flatten tensors to calculate the distance between every noise and image
        with torch.no_grad():
            x_0_flat = x_0.view(batch_size, -1)
            x_1_flat = x_1.view(batch_size, -1)
            
            # Compute a distance matrix (Batch_Size x Batch_Size)
            cost_matrix = torch.cdist(x_0_flat, x_1_flat).cpu().numpy()
            
            # Hungarian Algorithm finds the pairing that minimizes total distance
            _, col_ind = linear_sum_assignment(cost_matrix)
            
            # Reorder x_1 so it perfectly pairs with x_0 without crossing paths
            x_1 = x_1[col_ind]
        # --------------------------------------

        # 4. Sample continuous Time (t) from a Uniform Distribution [0, 1]
        t = torch.rand((batch_size,), device=device)
        
        # Reshape t to broadcast across image channels (B, 1, 1, 1)
        t_expand = t.view(-1, 1, 1, 1)
        
        # 5. Interpolate (The Straight Line Highway)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # 6. Calculate Ground Truth Target (Constant Velocity Vector)
        target_velocity = x_1 - x_0
        
        # 7. Network Prediction
        # Your U-Net takes x_t and t, and predicts the vector field
        pred_velocity = self.unet(x_t, t)
        
        # 8. Loss (MSE between predicted wind direction and true straight-line vector)
        return criterion(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_images(self, scheduler, latent_shape):
        self.unet.eval()
        device = next(self.unet.parameters()).device
        latents = torch.randn(latent_shape, device=device)
        
        for t in tqdm(reversed(range(scheduler.timesteps)), total=scheduler.timesteps, desc="Sampling", leave=False):
            t_batch = torch.full((latent_shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self(latents, t_batch)
            alpha_t = scheduler.extract(scheduler.alphas, t_batch, latents.shape)
            alpha_bar_t = scheduler.extract(scheduler.alphas_cumprod, t_batch, latents.shape)
            beta_t = scheduler.extract(scheduler.betas, t_batch, latents.shape)
            noise = torch.randn_like(latents) if t > 0 else torch.zeros_like(latents)
            latents = (1 / torch.sqrt(alpha_t)) * (latents - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * noise

        images = self.decode(latents)
        self.unet.train()
        return images
    
    @torch.no_grad()
    def sample_flow_images(self, latent_shape, num_steps=10):
        self.unet.eval()
        device = next(self.unet.parameters()).device
        
        # Start at t=0 (Pure Noise)
        x = torch.randn(latent_shape, device=device)
        
        # Create a timeline from 0 to 1
        t_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        
        for i in range(num_steps):
            # Current time and next time
            t_current = t_steps[i]
            t_next = t_steps[i + 1]
            step_size = t_next - t_current
            
            # Create a batched time tensor to feed the U-Net
            t_batch = torch.full((latent_shape[0],), t_current, device=device)
            
            # Predict the velocity vector at current position
            velocity = self.unet(x, t_batch)
            
            # Take a step forward along the straight line (Euler's Method)
            x = x + (velocity * step_size)
            
        # Decode the final latents (x_1) into pixels
        images = self.decode(x)
        self.unet.train()
        return images