import copy
import unittest

import torch

from dragn.config import load_config
from dragn.models.autoencoder_kl import AutoencoderKL, build_autoencoder
from dragn.training.autoencoder import AutoencoderKLLoss, LatentScaleEstimator


def tiny_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        latent_channels=2,
        base_channels=8,
        channel_multipliers=(1, 2, 2),
        num_res_blocks=1,
    )


class AutoencoderKLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shape_posterior_and_bounded_decoder(self):
        model = tiny_autoencoder()
        images = torch.randn(2, 3, 16, 16)
        output = model(images)

        self.assertEqual(output.posterior.mean.shape, (2, 2, 4, 4))
        self.assertEqual(output.posterior.logvar.shape, (2, 2, 4, 4))
        self.assertEqual(output.reconstruction.shape, images.shape)
        self.assertLessEqual(output.reconstruction.max().item(), 1.0)
        self.assertGreaterEqual(output.reconstruction.min().item(), -1.0)

    def test_smoke_yaml_builds_matching_eightfold_geometry(self):
        config = load_config("experiments/smoke_ae/config.yaml")
        model = build_autoencoder(config.autoencoder, config.data.channels)
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            posterior = model.encode(images)
        self.assertEqual(model.downsample_factor, 8)
        self.assertEqual(posterior.mean.shape, (1, 16, 8, 8))

    def test_reparameterized_path_has_gradients(self):
        model = tiny_autoencoder()
        images = torch.randn(2, 3, 16, 16)
        loss = model(images, sample_posterior=True).reconstruction.square().mean()
        loss.backward()

        self.assertIsNotNone(model.quant_conv.weight.grad)
        self.assertGreater(model.quant_conv.weight.grad.abs().sum().item(), 0.0)

    def test_kl_warmup(self):
        criterion = AutoencoderKLLoss(kl_weight=1e-6, kl_warmup_fraction=0.1)
        self.assertEqual(criterion.current_kl_weight(0, 100), 0.0)
        self.assertAlmostEqual(criterion.current_kl_weight(5, 100), 0.5e-6)
        self.assertAlmostEqual(criterion.current_kl_weight(10, 100), 1e-6)
        self.assertAlmostEqual(criterion.current_kl_weight(100, 100), 1e-6)

    def test_latent_scale_is_symmetric_and_serialized(self):
        model = tiny_autoencoder().eval()
        model.set_latent_scale(2.0)
        images = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            raw = model.encode(images).mode()
            scaled = model.encode_for_diffusion(images, sample_posterior=False)
            expected = model.decode(raw)
            actual = model.decode_from_diffusion(scaled)

        self.assertTrue(torch.allclose(scaled, raw * 0.5))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

        restored = tiny_autoencoder()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertEqual(restored.latent_scale.item(), 0.5)

    def test_streaming_latent_standard_deviation(self):
        estimator = LatentScaleEstimator()
        estimator.update(torch.tensor([0.0, 2.0]))
        self.assertAlmostEqual(estimator.standard_deviation().item(), 1.0)

    def test_tiny_batch_can_overfit(self):
        torch.manual_seed(7)
        model = tiny_autoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        images = torch.rand(2, 3, 16, 16) * 2.0 - 1.0

        def reconstruction_loss() -> torch.Tensor:
            output = model(images, sample_posterior=False)
            return torch.nn.functional.l1_loss(output.reconstruction, images)

        initial = reconstruction_loss().item()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss()
            loss.backward()
            optimizer.step()
        final = reconstruction_loss().item()

        self.assertLess(final, initial * 0.8)


if __name__ == "__main__":
    unittest.main()
