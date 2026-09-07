import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dragn.config import ExperimentConfig
from dragn.training.train_autoencoder import train_autoencoder
from dragn.training.train_generative import train_generative
from dragn.training.train_lora import train_lora


class GenerativeTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_data_ae_flow_dit_validation_and_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data" / "SDSS" / "spiral"
            data_dir.mkdir(parents=True)
            generator = np.random.default_rng(11)
            for index in range(8):
                pixels = generator.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(data_dir / f"{index}.png")

            shared = {
                "schema_version": 2,
                "experiment": {"output_dir": str(root / "results"), "seed": 4, "resume": "never"},
                "data": {
                    "dataset": "astro",
                    "root_dir": str(root / "data"),
                    "layout": "instrument_class",
                    "image_size": 16,
                    "channels": 3,
                    "transform": "astro_standard",
                    "batch_size": 2,
                    "num_workers": 0,
                    "validation_fraction": 0.25,
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
                    "epochs": 1,
                    "lr": 1e-3,
                    "precision": "fp32",
                    "max_steps": 2,
                    "validation_batches": 1,
                    "log_every": 1,
                },
            }
            ae_payload = {**shared, "task": "autoencoder"}
            ae_payload["experiment"] = {**shared["experiment"], "name": "test_ae"}
            ae_config = ExperimentConfig.model_validate(ae_payload)
            ae_checkpoint = train_autoencoder(ae_config)

            base_payload = {**shared, "task": "base"}
            base_payload["experiment"] = {**shared["experiment"], "name": "test_flow"}
            base_payload["autoencoder"] = {**shared["autoencoder"], "checkpoint": str(ae_checkpoint)}
            base_payload["model"] = {
                "patch_size": 2,
                "hidden_size": 32,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
            }
            base_payload["objective"] = {"type": "flow", "prediction": "velocity", "source_std": 1.0}
            base_payload["sampling"] = {"method": "euler", "steps": 2, "num_samples": 2}
            base_config = ExperimentConfig.model_validate(base_payload)
            checkpoint = train_generative(base_config)

            self.assertTrue(checkpoint.is_file())
            self.assertTrue((checkpoint.parent / "generated_epoch_0001.png").is_file())
            self.assertTrue((checkpoint.parent / "validation_reconstructions_epoch_0001.png").is_file())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["global_step"], 2)
            self.assertEqual(saved["objective"]["type"], "flow")
            self.assertIn("ema_state", saved)
            self.assertTrue((checkpoint.parent / "best.pt").is_file())

            lora_payload = {**base_payload, "task": "lora"}
            lora_payload["experiment"] = {
                **shared["experiment"], "name": "test_lora", "resume": "never",
            }
            lora_payload["data"] = {
                **base_payload["data"],
                "filter": {"instrument": "SDSS", "class_name": "spiral"},
            }
            lora_payload["lora"] = {
                "base_checkpoint": str(checkpoint),
                "base_weights": "ema",
                "preset": "full_lora",
                "rank": 4,
                "alpha": 4,
                "inference_scale": 1.0,
            }
            lora_checkpoint = train_lora(ExperimentConfig.model_validate(lora_payload))
            self.assertTrue(lora_checkpoint.is_file())
            self.assertTrue((lora_checkpoint.parent / "generated_epoch_0001.png").is_file())
            lora_saved = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(lora_saved["task"], "lora")
            self.assertEqual(lora_saved["preset"], "full_lora")
            self.assertNotIn("model_state", lora_saved)
            self.assertTrue(lora_saved["adapter_state"])
            self.assertTrue(all(
                name.endswith((".lora_a", ".lora_b"))
                for name in lora_saved["adapter_state"]
            ))
            self.assertTrue(any(
                torch.count_nonzero(value).item() > 0
                for name, value in lora_saved["adapter_state"].items()
                if name.endswith(".lora_b")
            ))
            lora_resume_payload = {**lora_payload}
            lora_resume_payload["experiment"] = {
                **lora_payload["experiment"], "resume": "required",
            }
            resumed_lora = train_lora(
                ExperimentConfig.model_validate(lora_resume_payload)
            )
            self.assertEqual(resumed_lora, lora_checkpoint)

            resume_payload = {**base_payload}
            resume_payload["experiment"] = {
                **base_payload["experiment"],
                "resume": "required",
            }
            resumed_checkpoint = train_generative(ExperimentConfig.model_validate(resume_payload))
            self.assertEqual(resumed_checkpoint, checkpoint)

            ddpm_payload = {**base_payload}
            ddpm_payload["experiment"] = {**shared["experiment"], "name": "test_ddpm"}
            ddpm_payload["objective"] = {
                "type": "ddpm",
                "prediction": "epsilon",
                "timesteps": 4,
                "schedule": "cosine",
            }
            ddpm_payload["sampling"] = {
                "method": "ancestral",
                "steps": 4,
                "num_samples": 2,
                "seed": 12,
            }
            ddpm_config = ExperimentConfig.model_validate(ddpm_payload)
            ddpm_checkpoint = train_generative(ddpm_config)
            self.assertTrue(ddpm_checkpoint.is_file())
            ddpm_saved = torch.load(ddpm_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(ddpm_saved["objective"]["type"], "ddpm")


if __name__ == "__main__":
    unittest.main()
