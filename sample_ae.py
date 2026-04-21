import argparse
import yaml
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Import our modular blocks
from autoenc import Autoencoder
from utils import get_device

def main():
    parser = argparse.ArgumentParser(description="Autoencoder Reconstruction Tester")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to your Stage 1 AE run directory")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image you want to test")
    parser.add_argument("--out", type=str, default="./results/ae_comparison.png", help="Where to save the side-by-side grid")
    parser.add_argument("--img_size", type=int, default=512, help="Image size expected by the model (e.g., 512 for Astro, 32 for MNIST)")
    args = parser.parse_args()

    dev = get_device()
    print(f"--- Booting AE Evaluator on {dev} ---")

    # 1. Load the Inference Blueprint
    run_dir = Path(args.run_dir)
    yaml_path = run_dir / "inference.yaml"
    weights_path = run_dir / "ae_weights.pth"
    
    if not yaml_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Could not find inference.yaml or ae_weights.pth in {run_dir}.")
        
    with open(yaml_path, 'r') as f: 
        inf_cfg = yaml.safe_load(f)

    # 2. Build the Architecture & Load Weights
    print("Building Autoencoder from blueprint...")
    ae = Autoencoder(inf_cfg['autoencoder']).to(dev)
    ae.load_state_dict(torch.load(weights_path, map_location=dev))
    ae.eval()

    # 3. Handle the Input Image
    print(f"Processing input image: {args.image}")
    
    # Check if model expects 1 channel (MNIST) or 3 channels (Astro)
    in_channels = inf_cfg['autoencoder']['channels'][0]
    img_mode = "L" if in_channels == 1 else "RGB"
    
    raw_img = Image.open(args.image).convert(img_mode)
    
    # Standard transform: Resize, ToTensor, Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * in_channels, [0.5] * in_channels)
    ])
    
    img_tensor = transform(raw_img).unsqueeze(0).to(dev) # Add batch dimension: (1, C, H, W)

    # 4. Forward Pass (Reconstruction)
    print("Extracting latents and decoding...")
    with torch.no_grad():
        recon_tensor, latents = ae(img_tensor)
        
    print(f"Latent Representation Shape: {latents.shape}")

    # 5. Format and Save Side-by-Side
    # Un-normalize from [-1, 1] to [0, 1] for saving
    img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
    recon_tensor = (recon_tensor.clamp(-1, 1) + 1) / 2

    # Concatenate the original and reconstruction horizontally
    # Shape becomes (2, C, H, W)
    comparison_grid = torch.cat([img_tensor, recon_tensor], dim=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # nrow=2 puts the two images side by side in the output file
    save_image(comparison_grid, out_path, nrow=2)
    print(f"Success! Side-by-side comparison saved to {out_path}")

if __name__ == "__main__":
    main()