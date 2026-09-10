import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dragn.classification.config import ClassificationConfig
from dragn.classification.data import read_manifest, write_split_manifest
from dragn.classification.metrics import classification_metrics


class ClassificationBenchmarkTests(unittest.TestCase):
    def test_joint_stratified_manifest_is_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data = root / "data"
            for instrument in ("SDSS", "SUBARU"):
                for object_name in ("lens", "spiral", "ring", "companion", "smooth"):
                    directory = data / instrument / object_name
                    directory.mkdir(parents=True)
                    for index in range(10):
                        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(directory / f"{index}.png")
            first = write_split_manifest(data, root / "first.csv", seed=42)
            second = write_split_manifest(data, root / "second.csv", seed=42)
            self.assertEqual(first.read_text(), second.read_text())
            rows = read_manifest(first, "real")
            self.assertEqual(len(rows), 100)
            self.assertEqual({row.split for row in rows}, {"train", "validation", "test"})
            self.assertEqual(len({row.path for row in rows}), len(rows))

    def test_task_scope_validation(self):
        payload = {
            "schema_version": 1, "task": "classification", "target": "object",
            "experiment": {"name": "test"}, "data": {"manifest": "split.csv"},
        }
        with self.assertRaises(ValueError):
            ClassificationConfig.model_validate(payload)

    def test_binary_metrics(self):
        metrics = classification_metrics(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1]),
            torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]]),
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIsNotNone(metrics["roc_auc"])


if __name__ == "__main__":
    unittest.main()
