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
from utils import get_device, save_images


def load_model_state(model, ckpt_path, dev):
    ckpt = torch.load(ckpt_path, map_location=dev)

    if isinstance(ckpt, dict):
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    return state_dict


def setup_workspace(yaml_path):
    with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
    out_dir = Path(config["output_dir"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    
    stage = config.get("stage", "bb")
    inf_config = {}
    
    ae_path = config.get("weights", {}).get("ae_base", "")
    if stage == "ae" or ae_path:
        inf_config["use_ae"] = True
        inf_config["autoencoder"] = config.get("autoencoder", {})
    else:
        inf_config["use_ae"] = False
        
    if stage in ["bb", "lora"]:
        inf_config["diffusion"] = config["diffusion"]
        inf_config["schedule"] = config["training"]["schedule"]
        inf_config["timesteps"] = config["training"]["timesteps"]
        
    if stage == "lora": 
        inf_config["lora_rank"] = config["lora"]["rank"]
        
    with open(out_dir / "inference.yaml", "w") as f: yaml.dump(inf_config, f, sort_keys=False)
    return config, out_dir

def train_autoencoder(ae, loader, cfg, out_dir, dev, auto_resume=True):
    opt = torch.optim.AdamW(ae.parameters(), lr=float(cfg["training"]["lr"]))
    start_epoch = 0
    ckpt_path = out_dir / "ae_checkpoint.pt"
    
    if auto_resume and ckpt_path.exists():
        print(f"\n[INFO] Auto-resuming AE training from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=dev)
        ae.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1

    fixed_images, _ = next(iter(loader))
    fixed_images = fixed_images[:8].to(dev)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
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
            ae.eval() 
            with torch.no_grad():
                fixed_recons, _ = ae(fixed_images)
            
            comparison_grid = torch.cat([fixed_images, fixed_recons], dim=0)
            save_path = out_dir / "samples" / f"ae_epoch_{epoch+1}.png"
            save_images(comparison_grid, save_path, nrow=8)
            
        torch.save({
            'epoch': epoch,
            'model_state': ae.state_dict(),
            'optimizer_state': opt.state_dict()
        }, ckpt_path)
        torch.save(ae.state_dict(), out_dir / "ae_weights.pth")

def train_backbone(ldm, loader, sched, crit, cfg, out_dir, dev, auto_resume=True):
    opt = torch.optim.AdamW(ldm.unet.parameters(), lr=float(cfg["training"]["lr"]))
    start_epoch = 0
    ckpt_path = out_dir / "bb_checkpoint.pt"
    
    if auto_resume and ckpt_path.exists():
        print(f"\n[INFO] Auto-resuming Backbone training from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=dev)
        ldm.unet.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
    
    sample_n = 16
    latent_size = cfg["diffusion"]["latent_size"]
    in_channels = cfg["diffusion"]["in_channels"]
    latent_shape = (sample_n, in_channels, latent_size, latent_size)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        ldm.train()
        pbar = tqdm(loader, desc=f"BB Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()
            loss = ldm.compute_loss(images, sched, crit)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % cfg["training"].get("sample_freq", 1) == 0:
            print(f"\nGenerating samples for epoch {epoch+1}...")
            samples = ldm.sample_images(sched, latent_shape)
            save_images(samples, out_dir / "samples" / f"epoch_{epoch+1}.png", nrow=4)

        torch.save({
            'epoch': epoch,
            'model_state': ldm.unet.state_dict(),
            'optimizer_state': opt.state_dict()
        }, ckpt_path)
        torch.save(ldm.unet.state_dict(), out_dir / "bb_weights.pth")

def train_flow_backbone(ldm, loader, crit, cfg, out_dir, dev, auto_resume=True):
    opt = torch.optim.AdamW(ldm.unet.parameters(), lr=float(cfg["training"]["lr"]))
    start_epoch = 0
    ckpt_path = out_dir / "flow_bb_checkpoint.pt"
    weights_path = out_dir / "flow_bb_weights.pth"

    if auto_resume and ckpt_path.exists():
        print(f"\n[INFO] Auto-resuming Flow Backbone training from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=dev)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            ldm.unet.load_state_dict(ckpt['model_state'])
            opt.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
        else:
            ldm.unet.load_state_dict(ckpt)
            start_epoch = 0
    elif auto_resume and weights_path.exists():
        print(f"\n[INFO] Auto-resuming legacy Flow Backbone weights from {weights_path}...")
        ldm.unet.load_state_dict(torch.load(weights_path, map_location=dev))
        start_epoch = 0

    sample_n = 16
    latent_size = cfg["diffusion"]["latent_size"]
    in_channels = cfg["diffusion"]["in_channels"]
    latent_shape = (sample_n, in_channels, latent_size, latent_size)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        ldm.train()
        pbar = tqdm(loader, desc=f"Flow BB Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()

            loss = ldm.compute_flow_loss(images, crit)

            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % cfg["training"].get("sample_freq", 1) == 0:
            print(f"\nGenerating Flow samples for epoch {epoch+1}...")
            samples = ldm.sample_flow_images(latent_shape, num_steps=cfg["training"]["timesteps"])
            save_images(samples, out_dir / "samples" / f"flow_epoch_{epoch+1}.png", nrow=4)

        torch.save({
            'epoch': epoch,
            'model_state': ldm.unet.state_dict(),
            'optimizer_state': opt.state_dict()
        }, ckpt_path)
        torch.save(ldm.unet.state_dict(), weights_path)

def train_lora(ldm, loader, sched, crit, cfg, out_dir, dev, auto_resume=True):
    params = [p for p in ldm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]))
    start_epoch = 0
    ckpt_path = out_dir / "lora_checkpoint.pt"
    
    if auto_resume and ckpt_path.exists():
        print(f"\n[INFO] Auto-resuming LoRA training from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=dev)
        ldm.unet.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
    
    sample_n = 16
    latent_size = cfg["diffusion"]["latent_size"]
    in_channels = cfg["diffusion"]["in_channels"]
    latent_shape = (sample_n, in_channels, latent_size, latent_size)

    is_flow = cfg["training"].get("type") == "flow"

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        ldm.train()

        if ldm.ae is not None:
            ldm.ae.eval()

        pbar = tqdm(loader, desc=f"LoRA Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()
            
            if is_flow:
                loss = ldm.compute_flow_loss(images, crit)
            else:
                loss = ldm.compute_loss(images, sched, crit)
                
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % cfg["training"].get("sample_freq", 1) == 0:
            print(f"\nGenerating samples for epoch {epoch+1}...")
            if is_flow:
                samples = ldm.sample_flow_images(latent_shape, num_steps=1000)
            else:
                samples = ldm.sample_images(sched, latent_shape)
                
            save_images(samples, out_dir / "samples" / f"epoch_{epoch+1}.png", nrow=4)

        torch.save({
            'epoch': epoch,
            'model_state': ldm.unet.state_dict(),
            'optimizer_state': opt.state_dict()
        }, ckpt_path)
        LoRAManager.save_weights(ldm.unet, out_dir / "lora_weights.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no_resume", action="store_true", help="Force a fresh restart by ignoring existing checkpoints")
    args = parser.parse_args()
    
    auto_resume = not args.no_resume
    dev = get_device()
    cfg, out_dir = setup_workspace(args.config)
    
    loader = DataFactory.create_loader(cfg)
    crit = LossFactory.get_loss(cfg["training"])
    ae_path = cfg.get("weights", {}).get("ae_base", "")

    if cfg["stage"] == "ae":
        ae = Autoencoder(cfg["autoencoder"]).to(dev)
        train_autoencoder(ae, loader, cfg, out_dir, dev, auto_resume)
        
    elif cfg["stage"] in ["bb", "lora"]:
        if ae_path:
            print(f"Loading Autoencoder from {ae_path}...")
            ae = Autoencoder(cfg["autoencoder"]).to(dev)
            load_model_state(ae, ae_path, dev)
            for p in ae.parameters(): p.requires_grad = False
            ae.eval()
        else:
            print("No AE path provided. Bypassing AE (Pixel-Space Diffusion).")
            ae = None
            
        bb = DiffusionTransformer(cfg["diffusion"]).to(dev)
        ldm = LatentDiffusionModel(ae, bb).to(dev)
        sched = DDPMScheduler(betas=getattr(NoiseSchedules, cfg["training"]["schedule"])(cfg["training"]["timesteps"]), device=dev)
        
        if cfg["stage"] == "bb":
            if cfg["training"].get("type") == "flow":
                print("Starting Optimal Transport Flow Matching Training...")
                train_flow_backbone(ldm, loader, crit, cfg, out_dir, dev, auto_resume)
            else:
                print("Starting Standard DDPM Training...")
                train_backbone(ldm, loader, sched, crit, cfg, out_dir, dev, auto_resume)
            
        elif cfg["stage"] == "lora":
            print(f"Loading backbone weights from {cfg['weights']['bb_base']}...")
            load_model_state(ldm.unet, cfg['weights']['bb_base'], dev)
            for p in ldm.unet.parameters(): p.requires_grad = False

            LoRAManager.inject_lora(ldm.unet, rank=cfg["lora"]["rank"], alpha=float(cfg["lora"]["rank"]))
            ldm.to(dev)

            train_lora(ldm, loader, sched, crit, cfg, out_dir, dev, auto_resume)

if __name__ == "__main__": main()