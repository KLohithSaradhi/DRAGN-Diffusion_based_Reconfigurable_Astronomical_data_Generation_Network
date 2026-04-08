import torch
import torch.nn as nn
from tqdm import tqdm

class LatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, unet):
        super().__init__()
        self.ae = autoencoder
        self.unet = unet
        for p in self.ae.parameters(): p.requires_grad = False
        self.ae.eval()

    @torch.no_grad()
    def encode(self, images): return self.ae.encoder(images)

    @torch.no_grad()
    def decode(self, latents): return self.ae.decoder(latents)

    def forward(self, noisy_latents, t): return self.unet(noisy_latents, t)

    def compute_loss(self, images, scheduler, criterion):
        latents = self.encode(images)
        device = latents.device
        t = torch.randint(0, scheduler.timesteps, (latents.shape[0],), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.q_sample(latents, t, noise)
        noise_pred = self(noisy_latents, t)
        return criterion(noise_pred, noise)

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