import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dragn.config import ExperimentConfig
from dragn.models import build_autoencoder
from dragn.training.train_autoencoder import train_autoencoder


class AutoencoderTrainingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_training_checkpoint_reload_and_inference_grid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data" / "SDSS" / "spiral"
            data_dir.mkdir(parents=True)
            generator = np.random.default_rng(5)
            for index in range(6):
                pixels = generator.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(data_dir / f"{index}.png")

            payload = {
                "schema_version": 2,
                "task": "autoencoder",
                "experiment": {
                    "name": "integration_smoke",
                    "output_dir": str(root / "results"),
                    "seed": 3,
                    "resume": "never",
                },
                "data": {
                    "dataset": "astro",
                    "root_dir": str(root / "data"),
                    "layout": "instrument_class",
                    "image_size": 16,
                    "channels": 3,
                    "transform": "astro_standard",
                    "batch_size": 2,
                    "num_workers": 0,
                    "validation_fraction": 0.34,
                },
                "autoencoder": {
                    "latent_channels": 2,
                    "downsample_factor": 4,
                    "base_channels": 8,
                    "channel_multipliers": [1, 2, 2],
                    "num_res_blocks": 1,
                    "loss": {"reconstruction": "l1", "kl_weight": 1e-6},
                },
                "training": {
                    "epochs": 2,
                    "lr": 1e-3,
                    "precision": "fp32",
                    "max_steps": 2,
                    "validation_batches": 1,
                    "log_every": 1,
                },
            }
            config = ExperimentConfig.model_validate(payload)
            checkpoint_path = train_autoencoder(config)
            grid_path = checkpoint_path.parent / "reconstructions.png"
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(grid_path.is_file())

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            reloaded = build_autoencoder(config.autoencoder, config.data.channels)
            reloaded.load_state_dict(checkpoint["model_state"])
            self.assertGreater(reloaded.latent_scale.item(), 0.0)
            self.assertEqual(checkpoint["step"], 2)


if __name__ == "__main__":
    unittest.main()
