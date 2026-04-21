import argparse
import yaml
import torch
import math
from pathlib import Path

# Import our modular blocks
from autoenc import Autoencoder
from transformer_diff import DiffusionTransformer
from model import LatentDiffusionModel
from diff import DDPMScheduler, NoiseSchedules
from lora import LoRAManager
from utils import get_device, save_images

def main():
    parser = argparse.ArgumentParser(description="Universal Inference Script for LDM")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the training run directory containing inference.yaml")
    parser.add_argument("--ae_weights", type=str, required=True, help="Path to frozen Autoencoder weights")
    parser.add_argument("--bb_weights", type=str, required=True, help="Path to Base DiT Backbone weights")
    parser.add_argument("--lora_weights", type=str, default=None, help="Optional: Path to LoRA weights (defaults to looking inside run_dir)")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--out", type=str, default="./results/samples.png", help="Path to save the generated grid")
    args = parser.parse_args()

    dev = get_device()
    print(f"--- Booting Generator on {dev} ---")

    # 1. Load the Inference Blueprint
    run_dir = Path(args.run_dir)
    yaml_path = run_dir / "inference.yaml"
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Could not find {yaml_path}. Are you sure this is a valid run directory?")
        
    with open(yaml_path, 'r') as f: 
        inf_cfg = yaml.safe_load(f)

    # Prevent sampling from an AE-only run
    if "diffusion" not in inf_cfg:
        raise ValueError("This inference.yaml does not contain diffusion parameters. Was this an Autoencoder-only run?")

    # 2. Build the Architecture
    print("Building architecture from blueprint...")
    ae = Autoencoder(inf_cfg['autoencoder'])
    bb = DiffusionTransformer(inf_cfg['diffusion'])
    model = LatentDiffusionModel(ae, bb).to(dev)
    
    # 3. Load Base Weights
    print(f"Loading AE weights: {args.ae_weights}")
    model.ae.load_state_dict(torch.load(args.ae_weights, map_location=dev), strict=False)
    
    print(f"Loading Base BB weights: {args.bb_weights}")
    model.unet.load_state_dict(torch.load(args.bb_weights, map_location=dev), strict=False)

    # 4. Handle LoRA Injection & Loading
    is_lora_run = 'lora_rank' in inf_cfg
    
    if is_lora_run:
        rank = inf_cfg['lora_rank']
        print(f"LoRA config detected! Injecting adapters (Rank {rank})...")
        LoRAManager.inject_lora(model.unet, rank=rank, alpha=float(rank))
        model.to(dev) # Push new adapter parameters to the GPU
        
        # Determine where to load LoRA weights from
        lora_path = args.lora_weights if args.lora_weights else (run_dir / "lora_weights.pth")
        if Path(lora_path).exists():
            print(f"Loading LoRA weights: {lora_path}")
            LoRAManager.load_weights(model.unet, lora_path, device=dev)
        else:
            print(f"WARNING: Expected LoRA weights at {lora_path} but file not found. Generating with empty adapters!")

    # 5. Setup Scheduler
    sched_type = inf_cfg.get("schedule", "cosine")
    timesteps = inf_cfg.get("timesteps", 1000)
    print(f"Setting up {sched_type} scheduler with {timesteps} steps...")
    
    betas = getattr(NoiseSchedules, sched_type)(timesteps)
    sched = DDPMScheduler(betas=betas, device=dev)

    # 6. Determine Latent Shape
    latent_shape = (
        args.n_samples, 
        inf_cfg["diffusion"]["in_channels"], 
        inf_cfg["diffusion"]["latent_size"], 
        inf_cfg["diffusion"]["latent_size"]
    )
    
    # 7. Generate!
    print(f"Generating {args.n_samples} images...")
    images = model.sample_images(sched, latent_shape)

    # 8. Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_rows = int(math.sqrt(args.n_samples))
    save_images(images, out_path, nrow=grid_rows)
    print(f"Success! Grid saved to {out_path}")

if __name__ == "__main__":
    main()