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
from utils import get_device

def setup_workspace(yaml_path):
    with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
    out_dir = Path(config["output_dir"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stage = config.get("stage", "bb")
    
    # Base inference config always has the Autoencoder
    inf_config = {
        "autoencoder": config["autoencoder"],
    }
    
    # Only add diffusion math if we are past stage 1
    if stage in ["bb", "lora"]:
        inf_config["diffusion"] = config["diffusion"]
        inf_config["schedule"] = config["training"]["schedule"]
        inf_config["timesteps"] = config["training"]["timesteps"]
        
    # Only add LoRA rank if we are doing LoRA
    if stage == "lora": 
        inf_config["lora_rank"] = config["lora"]["rank"]
        
    with open(out_dir / "inference.yaml", "w") as f: yaml.dump(inf_config, f, sort_keys=False)
    return config, out_dir

def train_autoencoder(ae, loader, cfg, out_dir, dev):
    opt = torch.optim.AdamW(ae.parameters(), lr=float(cfg["training"]["lr"]))
    for epoch in range(cfg["training"]["epochs"]):
        ae.train()
        pbar = tqdm(loader, desc=f"AE Ep {epoch+1}")
        for images, _ in pbar:
            images = images.to(dev)
            opt.zero_grad()
            recon, _ = ae(images)
            # You can also use crit here if you want huber/l1 for AE!
            loss = F.mse_loss(recon, images) 
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
        torch.save(ae.state_dict(), out_dir / "ae_weights.pth")

def train_backbone(ldm, loader, sched, crit, cfg, out_dir, dev):
    opt = torch.optim.AdamW(ldm.unet.parameters(), lr=float(cfg["training"]["lr"]))
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
        torch.save(ldm.unet.state_dict(), out_dir / "bb_weights.pth")

def train_lora(ldm, loader, sched, crit, cfg, out_dir, dev):
    params = [p for p in ldm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]))
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
        LoRAManager.save_weights(ldm.unet, out_dir / "lora_weights.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    dev = get_device()
    cfg, out_dir = setup_workspace(args.config)
    
    loader = DataFactory.create_loader(cfg)
    crit = LossFactory.get_loss(cfg["training"])

    # 1. ALWAYS initialize the Autoencoder
    ae = Autoencoder(cfg["autoencoder"]).to(dev)

    # 2. ROUTING LOGIC
    if cfg["stage"] == "ae":
        # Run AE without touching diffusion elements
        train_autoencoder(ae, loader, cfg, out_dir, dev)
        
    elif cfg["stage"] in ["bb", "lora"]:
        # We need diffusion components now!
        bb = DiffusionTransformer(cfg["diffusion"]).to(dev)
        ldm = LatentDiffusionModel(ae, bb).to(dev)
        sched = DDPMScheduler(betas=getattr(NoiseSchedules, cfg["training"]["schedule"])(cfg["training"]["timesteps"]), device=dev)
        
        if cfg["stage"] == "bb":
            ldm.ae.load_state_dict(torch.load(cfg['weights']['ae_base'], map_location=dev))
            train_backbone(ldm, loader, sched, crit, cfg, out_dir, dev)
            
        elif cfg["stage"] == "lora":
            ldm.ae.load_state_dict(torch.load(cfg['weights']['ae_base'], map_location=dev))
            ldm.unet.load_state_dict(torch.load(cfg['weights']['bb_base'], map_location=dev))
            for p in ldm.unet.parameters(): p.requires_grad = False
            
            LoRAManager.inject_lora(ldm.unet, rank=cfg["lora"]["rank"], alpha=float(cfg["lora"]["rank"]))
            ldm.to(dev)
            train_lora(ldm, loader, sched, crit, cfg, out_dir, dev)

if __name__ == "__main__": main()