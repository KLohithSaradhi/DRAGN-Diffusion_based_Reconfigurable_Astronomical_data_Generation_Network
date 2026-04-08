import argparse, yaml, torch
from pathlib import Path
from model import LatentDiffusionModel
from autoenc import Autoencoder
from transformer_diff import DiffusionTransformer
from diff import DDPMScheduler, NoiseSchedules
from lora import LoRAManager
from utils import get_device, save_images
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ae_weights", type=str, required=True)
    parser.add_argument("--bb_weights", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--out", type=str, default="./results/samples.png")
    args = parser.parse_args()
    dev = get_device()

    with open(Path(args.run_dir) / "inference.yaml", 'r') as f: inf_cfg = yaml.safe_load(f)
    
    ae = Autoencoder(inf_cfg['autoencoder'])
    bb = DiffusionTransformer(inf_cfg['diffusion'])
    model = LatentDiffusionModel(ae, bb).to(dev)
    
    model.ae.load_state_dict(torch.load(args.ae_weights, map_location=dev))
    model.unet.load_state_dict(torch.load(args.bb_weights, map_location=dev))

    if args.lora_weights and 'lora_rank' in inf_cfg:
        LoRAManager.inject_lora(model.unet, rank=inf_cfg['lora_rank'], alpha=float(inf_cfg['lora_rank']))
        model.to(dev)
        LoRAManager.load_weights(model.unet, args.lora_weights, device=dev)

    sched = DDPMScheduler(betas=getattr(NoiseSchedules, inf_cfg["schedule"])(inf_cfg["timesteps"]), device=dev)
    latent_shape = (args.n_samples, inf_cfg["diffusion"]["in_channels"], inf_cfg["diffusion"]["latent_size"], inf_cfg["diffusion"]["latent_size"])
    
    images = model.sample_images(sched, latent_shape)
    save_images(images, args.out, nrow=int(math.sqrt(args.n_samples)))

if __name__ == "__main__": main()