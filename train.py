import yaml, argparse, torch, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from data import DataFactory
from autoenc import Autoencoder
from transformer_diff import DiffusionTransformer
from model import LatentDiffusionModel
from diff import DDPMScheduler, NoiseSchedules
from lora import LoRAManager
from loss import LossFactory
from utils import get_device, save_images  # Added save_images

def setup_workspace(yaml_path):
    with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
    out_dir = Path(config["output_dir"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the samples directory
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    
    stage = config.get("stage", "bb")
    inf_config = {}
    
    # Check if we are actively using an Autoencoder (either training it, or loading its weights)
    ae_path = config.get("weights", {}).get("ae_base", "")
    if stage == "ae" or ae_path:
        inf_config["use_ae"] = True
        inf_config["autoencoder"] = config.get("autoencoder", {})
    else:
        # Bypassing the AE (Pixel-Space Diffusion)
        inf_config["use_ae"] = False
        
    if stage in ["bb", "lora"]:
        inf_config["diffusion"] = config["diffusion"]
        inf_config["schedule"] = config["training"]["schedule"]
        inf_config["timesteps"] = config["training"]["timesteps"]
        
    if stage == "lora": 
        inf_config["lora_rank"] = config["lora"]["rank"]
        
    with open(out_dir / "inference.yaml", "w") as f: yaml.dump(inf_config, f, sort_keys=False)
    return config, out_dir

def train_autoencoder(ae, loader, cfg, out_dir, dev):
    opt = torch.optim.AdamW(ae.parameters(), lr=float(cfg["training"]["lr"]))

    fixed_images, _ = next(iter(loader))
    fixed_images = fixed_images[:8].to(dev)


    for epoch in range(cfg["training"]["epochs"]):
        ae.train()
        pbar = tqdm(loader, desc=f"AE Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()
            recon, _ = ae(images)
            loss = F.mse_loss(recon, images) 
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
        if (epoch + 1) % cfg["training"].get("sample_freq", 1) == 0:
            print(f"\nGenerating AE reconstruction samples for epoch {epoch+1}...")
            ae.eval() # Turn off BatchNorm/Dropout for clean inference
            with torch.no_grad():
                fixed_recons, _ = ae(fixed_images)
            
            # Stack the original images on top of the reconstructions
            # Shape goes from (8, C, H, W) to (16, C, H, W)
            comparison_grid = torch.cat([fixed_images, fixed_recons], dim=0)
            
            # nrow=8 forces the layout: Top row is 8 originals, Bottom row is 8 recons
            save_path = out_dir / "samples" / f"ae_epoch_{epoch+1}.png"
            save_images(comparison_grid, save_path, nrow=8)
        torch.save(ae.state_dict(), out_dir / "ae_weights.pth")

def train_backbone(ldm, loader, sched, crit, cfg, out_dir, dev):
    opt = torch.optim.AdamW(ldm.unet.parameters(), lr=float(cfg["training"]["lr"]))
    
    # Define sampling shape (e.g., a 4x4 grid)
    sample_n = 16
    latent_size = cfg["diffusion"]["latent_size"]
    in_channels = cfg["diffusion"]["in_channels"]
    latent_shape = (sample_n, in_channels, latent_size, latent_size)

    for epoch in range(cfg["training"]["epochs"]):
        ldm.train()
        pbar = tqdm(loader, desc=f"BB Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()
            loss = ldm.compute_loss(images, sched, crit)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        # Periodically sample images
        if (epoch + 1) % cfg["training"].get("sample_freq", 1) == 0:
            print(f"\nGenerating samples for epoch {epoch+1}...")
            samples = ldm.sample_images(sched, latent_shape)
            save_images(samples, out_dir / "samples" / f"epoch_{epoch+1}.png", nrow=4)

        torch.save(ldm.unet.state_dict(), out_dir / "bb_weights.pth")

def train_lora(ldm, loader, sched, crit, cfg, out_dir, dev):
    params = [p for p in ldm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]))
    
    # Define sampling shape
    sample_n = 16
    latent_size = cfg["diffusion"]["latent_size"]
    in_channels = cfg["diffusion"]["in_channels"]
    latent_shape = (sample_n, in_channels, latent_size, latent_size)

    for epoch in range(cfg["training"]["epochs"]):
        ldm.train()
        pbar = tqdm(loader, desc=f"LoRA Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()
            loss = ldm.compute_loss(images, sched, crit)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        # Periodically sample images
        if (epoch + 1) % cfg["training"].get("sample_freq", 1) == 0:
            print(f"\nGenerating samples for epoch {epoch+1}...")
            samples = ldm.sample_images(sched, latent_shape)
            save_images(samples, out_dir / "samples" / f"epoch_{epoch+1}.png", nrow=4)

        LoRAManager.save_weights(ldm.unet, out_dir / "lora_weights.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    dev = get_device()
    cfg, out_dir = setup_workspace(args.config)
    
    loader = DataFactory.create_loader(cfg)
    crit = LossFactory.get_loss(cfg["training"])

    ae_path = cfg.get("weights", {}).get("ae_base", "")

    

    if cfg["stage"] == "ae":
        ae = Autoencoder(cfg["autoencoder"]).to(dev)
        # Run AE without touching diffusion elements
        train_autoencoder(ae, loader, cfg, out_dir, dev)
        
    elif cfg["stage"] in ["bb", "lora"]:
        if ae_path:
            print(f"Loading Autoencoder from {ae_path}...")
            ae = Autoencoder(cfg["autoencoder"]).to(dev)
            ae.load_state_dict(torch.load(ae_path, map_location=dev))
            # CRITICAL: Freeze the AE so BB training doesn't destroy its weights!
            for p in ae.parameters(): p.requires_grad = False
            ae.eval()
        else:
            print("No AE path provided. Bypassing AE (Pixel-Space Diffusion).")
            ae = None
        # We need diffusion components now!
        bb = DiffusionTransformer(cfg["diffusion"]).to(dev)
        ldm = LatentDiffusionModel(ae, bb).to(dev)
        sched = DDPMScheduler(betas=getattr(NoiseSchedules, cfg["training"]["schedule"])(cfg["training"]["timesteps"]), device=dev)
        
        if cfg["stage"] == "bb":
            train_backbone(ldm, loader, sched, crit, cfg, out_dir, dev)
            
        elif cfg["stage"] == "lora":
            ldm.unet.load_state_dict(torch.load(cfg['weights']['bb_base'], map_location=dev))
            for p in ldm.unet.parameters(): p.requires_grad = False
            
            LoRAManager.inject_lora(ldm.unet, rank=cfg["lora"]["rank"], alpha=float(cfg["lora"]["rank"]))
            ldm.to(dev)
            train_lora(ldm, loader, sched, crit, cfg, out_dir, dev)

if __name__ == "__main__": main()