"""Dependency-light image grid output."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor


def save_image_grid(images: Tensor, path: str | Path, columns: int) -> None:
    images = images.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte()
    count, channels, height, width = images.shape
    rows = (count + columns - 1) // columns
    grid = np.zeros((rows * height, columns * width, channels), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        grid[row * height:(row + 1) * height, column * width:(column + 1) * width] = image.permute(1, 2, 0).numpy()
    if channels == 1:
        grid = grid[:, :, 0]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
