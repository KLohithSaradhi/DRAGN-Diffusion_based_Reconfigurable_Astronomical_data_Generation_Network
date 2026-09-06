import yaml
import argparse
from pathlib import Path
from data import DataFactory
from utils import save_images

def main():
    parser = argparse.ArgumentParser(description="Visualize training data samples from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--out", type=str, default="sample_grid.png", help="Output path for the grid image.")
    args = parser.parse_args()

    # 1. Load the YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading dataset based on {args.config}...")
    
    # 2. Create the dataloader using your existing factory
    loader = DataFactory.create_loader(config)

    # 3. Fetch a single batch
    print("Fetching a batch of data...")
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # 4. We want a 6x6 grid (36 images)
    num_samples = 36
    
    if images.shape[0] < num_samples:
        print(f"Warning: Batch size ({images.shape[0]}) is less than 36. Grid will be smaller.")
        num_samples = images.shape[0]

    grid_images = images[:num_samples]

    # 5. Save the grid
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Your utils.py handles the denormalization automatically!
    save_images(grid_images, str(out_path), nrow=6)
    
    print(f"Successfully saved {num_samples} samples in a 6x6 grid to {out_path}")

if __name__ == "__main__":
    main()