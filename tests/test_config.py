import unittest

from pydantic import ValidationError

from dragn.config import ExperimentConfig, InferenceConfig


def autoencoder_payload() -> dict:
    return {
        "schema_version": 2,
        "task": "autoencoder",
        "experiment": {"name": "smoke_ae", "resume": "never"},
        "data": {
            "dataset": "astro",
            "root_dir": "./data/smoke_astro",
            "layout": "instrument_class",
            "image_size": 64,
            "channels": 3,
            "transform": "astro_standard",
            "batch_size": 2,
            "num_workers": 0,
        },
        "autoencoder": {
            "latent_channels": 16,
            "downsample_factor": 8,
            "base_channels": 32,
            "channel_multipliers": [1, 2, 4, 4],
            "loss": {"reconstruction": "l1", "kl_weight": 1e-6},
        },
        "training": {"epochs": 2, "lr": 1e-4, "precision": "fp32"},
    }


def lora_payload() -> dict:
    payload = autoencoder_payload()
    payload["task"] = "lora"
    payload["experiment"]["name"] = "spiral_sdss_lora"
    payload["data"]["filter"] = {"instrument": "SDSS", "class_name": "spiral"}
    payload["autoencoder"]["checkpoint"] = "./results/smoke_ae/best.pt"
    payload["model"] = {
        "patch_size": 2,
        "hidden_size": 256,
        "depth": 6,
        "num_heads": 8,
        "positional_embedding": "sincos_2d",
    }
    payload["objective"] = {"type": "flow", "prediction": "velocity", "source_std": 1.0}
    payload["sampling"] = {"method": "euler", "steps": 20}
    payload["lora"] = {
        "base_checkpoint": "./results/smoke_flow/best.pt",
        "base_weights": "ema",
        "preset": "proj_lora",
        "rank": 8,
        "alpha": 8,
    }
    return payload


class ConfigTests(unittest.TestCase):
    def test_autoencoder_config_is_valid(self):
        config = ExperimentConfig.model_validate(autoencoder_payload())
        self.assertEqual(config.autoencoder.latent_channels, 16)

    def test_joint_lora_config_is_valid(self):
        config = ExperimentConfig.model_validate(lora_payload())
        self.assertEqual(config.data.filter.instrument, "SDSS")
        self.assertEqual(config.data.filter.class_name, "spiral")
        self.assertEqual(config.lora.targets, ("attention.proj", "mlp.fc1", "mlp.fc2"))

    def test_unknown_keys_are_rejected(self):
        payload = autoencoder_payload()
        payload["training"]["mystery_option"] = True
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(payload)

    def test_invalid_latent_geometry_is_rejected(self):
        payload = autoencoder_payload()
        payload["data"]["image_size"] = 65
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(payload)

    def test_objective_and_sampler_must_match(self):
        payload = lora_payload()
        payload["sampling"]["method"] = "ancestral"
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(payload)

    def test_ddpm_ancestral_steps_must_match_training_timesteps(self):
        payload = lora_payload()
        payload["objective"] = {"type": "ddpm", "prediction": "epsilon", "timesteps": 10}
        payload["sampling"] = {"method": "ancestral", "steps": 5}
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(payload)

    def test_lora_requires_joint_filter(self):
        payload = lora_payload()
        payload["data"]["filter"] = None
        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(payload)

    def test_inference_adapter_names_must_be_unique(self):
        payload = {
            "schema_version": 2,
            "task": "inference",
            "experiment": {"name": "compare"},
            "inference": {
                "autoencoder_checkpoint": "ae.pt",
                "base_checkpoint": "base.pt",
                "adapters": [
                    {"name": "spiral", "checkpoint": "one.pt"},
                    {"name": "spiral", "checkpoint": "two.pt"},
                ],
            },
            "sampling": {"method": "euler", "steps": 10},
        }
        with self.assertRaises(ValidationError):
            InferenceConfig.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
