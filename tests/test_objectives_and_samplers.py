import unittest

import torch

from dragn.models import DiT
from dragn.objectives import DiffusionSchedule, ddpm_prediction_and_target, flow_prediction_and_target
from dragn.samplers import sample_ddpm, sample_flow, seeded_generator


class ObjectiveAndSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flow_training_and_sampling_shapes(self):
        model = DiT(2, 8, patch_size=2, hidden_size=32, depth=1, num_heads=4)
        latents = torch.randn(3, 2, 8, 8)
        prediction, target = flow_prediction_and_target(
            model, latents, 1.0, seeded_generator(torch.device("cpu"), 4)
        )
        self.assertEqual(prediction.shape, latents.shape)
        self.assertEqual(target.shape, latents.shape)
        sampled = sample_flow(
            model,
            (2, 2, 8, 8),
            1.0,
            2,
            seeded_generator(torch.device("cpu"), 5),
        )
        self.assertEqual(sampled.shape, (2, 2, 8, 8))

    def test_ddpm_training_and_sampling_shapes(self):
        model = DiT(2, 8, patch_size=2, hidden_size=32, depth=1, num_heads=4)
        schedule = DiffusionSchedule.create(4, "cosine", torch.device("cpu"))
        latents = torch.randn(3, 2, 8, 8)
        prediction, target = ddpm_prediction_and_target(
            model, latents, schedule, seeded_generator(torch.device("cpu"), 6)
        )
        self.assertEqual(prediction.shape, latents.shape)
        self.assertEqual(target.shape, latents.shape)
        sampled = sample_ddpm(
            model,
            (2, 2, 8, 8),
            schedule,
            seeded_generator(torch.device("cpu"), 7),
        )
        self.assertEqual(sampled.shape, (2, 2, 8, 8))

    def test_posterior_variance_is_zero_at_final_denoising_step(self):
        schedule = DiffusionSchedule.create(10, "linear", torch.device("cpu"))
        self.assertEqual(schedule.posterior_variance[0].item(), 0.0)
        self.assertTrue(torch.all(schedule.posterior_variance[1:] > 0))

    def test_sampling_seed_is_reproducible(self):
        model = DiT(1, 4, patch_size=2, hidden_size=32, depth=1, num_heads=4)
        first = sample_flow(
            model, (1, 1, 4, 4), 1.0, 2, seeded_generator(torch.device("cpu"), 9)
        )
        second = sample_flow(
            model, (1, 1, 4, 4), 1.0, 2, seeded_generator(torch.device("cpu"), 9)
        )
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
