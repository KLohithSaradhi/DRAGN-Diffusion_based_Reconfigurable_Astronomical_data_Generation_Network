import tempfile
import unittest
from pathlib import Path

from dragn.config import ExperimentConfig, InferenceConfig, load_config
from dragn.training.inference_bundle import ensure_inference_bundle


def base_payload(root: Path) -> dict:
    return {
        "schema_version": 2,
        "task": "base",
        "experiment": {
            "name": "base_run", "output_dir": str(root / "results"),
            "seed": 7, "resume": "auto",
        },
        "data": {
            "dataset": "astro", "root_dir": str(root / "data"),
            "layout": "instrument_class", "image_size": 16, "channels": 3,
            "transform": "astro_standard", "batch_size": 2,
        },
        "autoencoder": {
            "checkpoint": str(root / "ae.pt"), "latent_channels": 2,
            "downsample_factor": 4, "base_channels": 8,
            "channel_multipliers": [1, 2, 2],
        },
        "model": {
            "patch_size": 2, "hidden_size": 32, "depth": 1, "num_heads": 4,
        },
        "objective": {"type": "flow", "prediction": "velocity"},
        "training": {"epochs": 1, "lr": 1e-4, "ema_decay": 0.9999},
        "sampling": {"method": "euler", "steps": 2, "num_samples": 2},
    }


class InferenceBundleTests(unittest.TestCase):
    def test_base_bundle_is_created_once_and_targets_nested_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = ExperimentConfig.model_validate(base_payload(root))
            experiment_directory = root / "experiments" / "base_run"
            latest = root / "results" / "base_run" / "latest.pt"
            yaml_path, sbatch_path = ensure_inference_bundle(
                config, latest, experiment_directory
            )
            generated = load_config(yaml_path)
            self.assertIsInstance(generated, InferenceConfig)
            self.assertEqual(
                generated.experiment.output_dir / generated.experiment.name,
                root / "results" / "base_run" / "inference",
            )
            self.assertEqual(generated.inference.base_checkpoint, latest)
            self.assertEqual(generated.inference.base_weights, "ema")
            self.assertEqual(generated.inference.adapters, [])
            self.assertIn("#SBATCH --output=logs/%x_%j.out", sbatch_path.read_text())
            self.assertTrue((experiment_directory / "logs").is_dir())

            yaml_path.write_text("do-not-rewrite\n", encoding="utf-8")
            sbatch_path.write_text("do-not-rewrite\n", encoding="utf-8")
            ensure_inference_bundle(config, latest, experiment_directory)
            self.assertEqual(yaml_path.read_text(), "do-not-rewrite\n")
            self.assertEqual(sbatch_path.read_text(), "do-not-rewrite\n")

    def test_lora_bundle_selects_only_its_adapter_and_weight_variants(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = base_payload(root)
            payload["task"] = "lora"
            payload["experiment"]["name"] = "lens_adapter"
            payload["data"]["filter"] = {
                "instrument": "SDSS", "class_name": "lens",
            }
            payload["training"]["ema_decay"] = None
            payload["lora"] = {
                "base_checkpoint": str(root / "base.pt"),
                "base_weights": "raw", "preset": "mlp_lora",
                "rank": 8, "alpha": 8, "inference_scale": 1.5,
            }
            config = ExperimentConfig.model_validate(payload)
            latest = root / "results" / "lens_adapter" / "latest.pt"
            yaml_path, _ = ensure_inference_bundle(
                config, latest, root / "experiments" / "lens_adapter"
            )
            generated = load_config(yaml_path)
            self.assertIsInstance(generated, InferenceConfig)
            self.assertEqual(generated.inference.base_checkpoint, root / "base.pt")
            self.assertEqual(generated.inference.base_weights, "raw")
            self.assertEqual(len(generated.inference.adapters), 1)
            adapter = generated.inference.adapters[0]
            self.assertEqual(adapter.name, "mlp_lora")
            self.assertEqual(adapter.checkpoint, latest)
            self.assertEqual(adapter.weights, "raw")
            self.assertEqual(adapter.scale, 1.5)


if __name__ == "__main__":
    unittest.main()
