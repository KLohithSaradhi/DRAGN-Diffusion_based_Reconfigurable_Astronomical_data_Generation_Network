import unittest

import torch

from dragn.models.dit import DiT, FourierTimeEmbedding, fixed_2d_sincos_position_embedding


class DiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_position_embedding_shape_and_determinism(self):
        first = fixed_2d_sincos_position_embedding(4, 6, 32)
        second = fixed_2d_sincos_position_embedding(4, 6, 32)
        self.assertEqual(first.shape, (1, 24, 32))
        self.assertTrue(torch.equal(first, second))

    def test_rectangular_forward_preserves_shape_and_starts_at_zero(self):
        model = DiT(3, (8, 12), patch_size=2, hidden_size=32, depth=2, num_heads=4)
        inputs = torch.randn(2, 3, 8, 12)
        output = model(inputs, torch.tensor([0.0, 1.0]))
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertNotIn("position_embedding", dict(model.named_parameters()))

    def test_output_projection_receives_gradient(self):
        model = DiT(2, 8, patch_size=2, hidden_size=32, depth=2, num_heads=4)
        output = model(torch.randn(2, 2, 8, 8), torch.tensor([0.25, 0.75]))
        output.sum().backward()
        gradient = model.final_layer.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_time_embedding_covers_normalized_interval(self):
        embedding = FourierTimeEmbedding(32)
        values = embedding(torch.tensor([0.0, 0.5, 1.0]))
        self.assertEqual(values.shape, (3, 32))
        self.assertFalse(torch.equal(values[0], values[1]))
        self.assertFalse(torch.equal(values[1], values[2]))

    def test_time_outside_unit_interval_is_rejected(self):
        model = DiT(2, 8, patch_size=2, hidden_size=32, depth=1, num_heads=4)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 8, 8), torch.tensor([2.0]))

    def test_wrong_spatial_shape_is_rejected(self):
        model = DiT(2, 8, patch_size=2, hidden_size=32, depth=1, num_heads=4)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 10, 8), torch.tensor([0.5]))


if __name__ == "__main__":
    unittest.main()
