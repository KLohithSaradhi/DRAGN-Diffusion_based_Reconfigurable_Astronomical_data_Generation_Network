import tempfile
import unittest
from pathlib import Path

import torch

from dragn.config import ExperimentConfig
from dragn.training.common import (
    CheckpointManager,
    ExponentialMovingAverage,
    capture_rng_state,
    experiment_signature,
    restore_rng_state,
)


def base_payload() -> dict:
    return {
        "schema_version": 2,
        "task": "base",
        "experiment": {"name": "resume_test", "resume": "auto"},
        "data": {
            "dataset": "astro",
            "root_dir": "./data",
            "layout": "instrument_class",
            "image_size": 16,
            "channels": 3,
            "transform": "astro_standard",
            "batch_size": 2,
        },
        "autoencoder": {
            "checkpoint": "./ae.pt",
            "latent_channels": 2,
            "downsample_factor": 4,
            "base_channels": 8,
            "channel_multipliers": [1, 2, 2],
        },
        "model": {"patch_size": 2, "hidden_size": 32, "depth": 1, "num_heads": 4},
        "objective": {"type": "flow", "prediction": "velocity"},
        "training": {"epochs": 3, "lr": 1e-4},
        "sampling": {"method": "euler", "steps": 2},
    }


class TrainingCommonTests(unittest.TestCase):
    def test_rng_state_restores_cpu_byte_tensor(self):
        torch.manual_seed(123)
        state = capture_rng_state()
        expected = torch.rand(4)
        torch.rand(8)
        restore_rng_state(state)
        torch.testing.assert_close(torch.rand(4), expected)
        self.assertEqual(state["torch"].device, torch.device("cpu"))
        self.assertEqual(state["torch"].dtype, torch.uint8)

    def test_ema_updates_without_aliasing_training_model(self):
        model = torch.nn.Linear(2, 2, bias=False)
        ema = ExponentialMovingAverage(model, decay=0.5)
        original = ema.model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(2.0)
        ema.update(model)
        self.assertTrue(torch.allclose(ema.model.weight, original + 1.0))
        self.assertFalse(ema.model.weight.requires_grad)

    def test_checkpoint_rotation_and_strict_signature(self):
        config = ExperimentConfig.model_validate(base_payload())
        signature = experiment_signature(config, "ae-hash")
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CheckpointManager(Path(temporary_directory), keep=3)
            for epoch in range(1, 6):
                manager.save({"signature": signature, "epoch": epoch}, epoch, is_best=epoch == 2)
            epochs = sorted(manager.output_dir.glob("checkpoint_epoch_*.pt"))
            self.assertEqual([path.name for path in epochs], [
                "checkpoint_epoch_0003.pt",
                "checkpoint_epoch_0004.pt",
                "checkpoint_epoch_0005.pt",
            ])
            loaded = manager.load_for_resume("required", signature, torch.device("cpu"))
            self.assertEqual(loaded["epoch"], 5)
            with self.assertRaises(ValueError):
                manager.load_for_resume("auto", "wrong-signature", torch.device("cpu"))

    def test_latest_can_be_saved_without_archiving_an_epoch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CheckpointManager(Path(temporary_directory), keep=3)
            manager.save_latest({"epoch": 1})
            self.assertTrue(manager.latest_path.is_file())
            self.assertEqual(list(manager.output_dir.glob("checkpoint_epoch_*.pt")), [])

    def test_architecture_change_changes_signature(self):
        first_payload = base_payload()
        second_payload = base_payload()
        second_payload["model"] = {**second_payload["model"], "depth": 2}
        first = experiment_signature(ExperimentConfig.model_validate(first_payload), "same-ae")
        second = experiment_signature(ExperimentConfig.model_validate(second_payload), "same-ae")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
