import torch
import yaml
import argparse
import gc
from pathlib import Path
from tqdm import tqdm

from transformer_diff import DiffusionTransformer
from model import LatentDiffusionModel
from lora import LoRAManager
from utils import get_device, save_images

def get_lora_path(cfg):
    """Dynamically extracts the LoRA path from the config exactly like train.py saves it."""
    # If you decide to explicitly pass it in the weights dictionary
    if "weights" in cfg and "lora" in cfg["weights"] and cfg["weights"]["lora"]:
        return Path(cfg["weights"]["lora"])
    # Otherwise, default to the directory structure train.py creates
    return Path(cfg["output_dir"]) / cfg["run_name"] / "lora_weights.pth"

def build_model_with_lora(base_cfg, lora_cfg, bb_weights_path, dev):
    """Builds a fresh base U-Net and injects the specific LoRA configuration."""
    print(f"-> Initializing fresh Backbone U-Net...")
    bb = DiffusionTransformer(base_cfg["diffusion"]).to(dev)
    ldm = LatentDiffusionModel(autoencoder=None, unet=bb).to(dev)
    
    print(f"-> Loading base weights...")
    ldm.unet.load_state_dict(torch.load(bb_weights_path, map_location=dev))
    for p in ldm.unet.parameters(): p.requires_grad = False
    
    rank = lora_cfg["lora"]["rank"]
    alpha = float(lora_cfg["lora"].get("alpha", rank))  # Fallback to rank if alpha not in yaml
    
    print(f"-> Injecting LoRA architecture (Rank={rank}, Alpha={alpha})...")
    LoRAManager.inject_lora(ldm.unet, rank=rank, alpha=alpha)
    ldm.to(dev)
    
    return ldm

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Zero-shot Config-Driven Translation")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base flow_template.yaml")
    parser.add_argument("--bb_weights", type=str, required=True, help="Path to base backbone weights")
    parser.add_argument("--obj_config", type=str, required=True, help="Path to Object LoRA yaml")
    parser.add_argument("--inst_config", type=str, required=True, help="Path to Instrument LoRA yaml")
    parser.add_argument("--out_dir", type=str, default="./results/translation", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, default=20, help="Total ODE integration steps")
    parser.add_argument("--degrade_steps", type=int, default=8, help="How many steps to slide back on ODE")
    args = parser.parse_args()

    dev = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Configs
    with open(args.base_config, 'r') as f: base_cfg = yaml.safe_load(f)
    with open(args.obj_config, 'r') as f: obj_cfg = yaml.safe_load(f)
    with open(args.inst_config, 'r') as f: inst_cfg = yaml.safe_load(f)

    latent_size = base_cfg["diffusion"]["latent_size"]
    in_channels = base_cfg["diffusion"]["in_channels"]
    latent_shape = (args.num_samples, in_channels, latent_size, latent_size)

    # ==========================================
    # PHASE 1: GENERATE PRISTINE OBJECT
    # ==========================================
    print("\n=== PHASE 1: OBJECT GENERATION ===")
    ldm = build_model_with_lora(base_cfg, obj_cfg, args.bb_weights, dev)
    
    obj_lora_path = get_lora_path(obj_cfg)
    print(f"Loading Object LoRA weights from {obj_lora_path}...")
    LoRAManager.load_weights(ldm.unet, obj_lora_path, dev)
    ldm.unet.eval()

    x_obj = torch.randn(latent_shape, device=dev)
    t_steps = torch.linspace(0, 1.0, args.num_steps + 1, device=dev)
    
    print("Integrating ODE for object morphology...")
    for i in tqdm(range(args.num_steps), desc="Object Flow", leave=False):
        t_current = t_steps[i]
        t_next = t_steps[i + 1]
        step_size = t_next - t_current
        
        t_batch = torch.full((latent_shape[0],), t_current, device=dev)
        velocity = ldm.unet(x_obj, t_batch)
        x_obj = x_obj + (velocity * step_size)
        
    clean_images = ldm.decode(x_obj)

    # Clean up Phase 1 model to free VRAM for Phase 2
    del ldm
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # PHASE 2: STRAIGHT-LINE DEGRADATION
    # ==========================================
    t_start_idx = args.num_steps - args.degrade_steps
    t_start = t_steps[t_start_idx].item()
    
    print(f"\n=== PHASE 2: DEGRADATION ===")
    print(f"Sliding back {args.degrade_steps} steps to t={t_start:.2f}...")
    fresh_noise = torch.randn_like(x_obj)
    x_degraded = (1 - t_start) * fresh_noise + t_start * x_obj

    # ==========================================
    # PHASE 3: RECONSTRUCT INSTRUMENT
    # ==========================================
    print("\n=== PHASE 3: INSTRUMENT RENDERING ===")
    ldm = build_model_with_lora(base_cfg, inst_cfg, args.bb_weights, dev)
    
    inst_lora_path = get_lora_path(inst_cfg)
    print(f"Loading Instrument LoRA weights from {inst_lora_path}...")
    LoRAManager.load_weights(ldm.unet, inst_lora_path, dev)
    ldm.unet.eval()
    
    t_steps_inst = torch.linspace(t_start, 1.0, args.degrade_steps + 1, device=dev)
    x_inst = x_degraded
    
    print("Integrating ODE for instrument dependence...")
    for i in tqdm(range(args.degrade_steps), desc="Instrument Flow", leave=False):
        t_current = t_steps_inst[i]
        t_next = t_steps_inst[i + 1]
        step_size = t_next - t_current
        
        t_batch = torch.full((latent_shape[0],), t_current, device=dev)
        velocity = ldm.unet(x_inst, t_batch) 
        x_inst = x_inst + (velocity * step_size)
        
    inst_images = ldm.decode(x_inst)

    # ==========================================
    # PHASE 4: SAVING OUTPUTS
    # ==========================================
    print("\n=== PHASE 4: SAVING ===")
    comparison_grid = torch.empty((args.num_samples * 2, *clean_images.shape[1:]), device=dev)
    comparison_grid[0::2] = clean_images
    comparison_grid[1::2] = inst_images
    
    save_path = out_dir / "config_driven_translation.png"
    save_images(comparison_grid, save_path, nrow=8)
    print(f"Translation complete! Grid saved to {save_path}")

if __name__ == "__main__":
    main()