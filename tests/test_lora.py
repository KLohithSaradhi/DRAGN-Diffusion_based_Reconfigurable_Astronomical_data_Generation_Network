import unittest

import torch

from dragn.config import LoRASection
from dragn.models import (
    AdapterEMA,
    LoRALinear,
    adapter_parameters,
    adapter_state_dict,
    inject_lora,
    load_adapter_state_dict,
)
from dragn.models.dit import DiT


class LoRATests(unittest.TestCase):
    def _model(self):
        return DiT(
            in_channels=2, input_size=4, patch_size=2, hidden_size=32,
            depth=2, num_heads=4, mlp_ratio=2.0,
        )

    def test_presets_replace_exact_requested_modules(self):
        expected_per_block = {"full_lora": 4, "proj_lora": 3, "mlp_lora": 2}
        for preset, count in expected_per_block.items():
            with self.subTest(preset=preset):
                model = self._model()
                config = LoRASection(
                    base_checkpoint="base.pt", preset=preset, rank=4, alpha=4,
                )
                replaced = inject_lora(model, config)
                self.assertEqual(len(replaced), count * len(model.blocks))
                self.assertTrue(all(parameter.requires_grad for parameter in adapter_parameters(model)))
                self.assertTrue(all(
                    parameter.requires_grad == (name.endswith(".lora_a") or name.endswith(".lora_b"))
                    for name, parameter in model.named_parameters()
                ))

    def test_zero_initialized_adapter_preserves_base_output(self):
        model = self._model().eval()
        inputs = torch.randn(2, 2, 4, 4)
        time = torch.rand(2)
        expected = model(inputs, time)
        inject_lora(
            model,
            LoRASection(base_checkpoint="base.pt", preset="full_lora", rank=4, alpha=4),
        )
        torch.testing.assert_close(model(inputs, time), expected)

    def test_adapter_state_and_ema_are_adapter_only(self):
        model = self._model()
        inject_lora(
            model,
            LoRASection(base_checkpoint="base.pt", preset="mlp_lora", rank=4, alpha=4),
        )
        state = adapter_state_dict(model)
        self.assertTrue(state)
        self.assertTrue(all(name.endswith((".lora_a", ".lora_b")) for name in state))
        ema = AdapterEMA(model, 0.9)
        with torch.no_grad():
            next(adapter_parameters(model)).add_(1)
        raw = adapter_state_dict(model)
        with ema.apply(model):
            self.assertFalse(torch.equal(next(iter(adapter_state_dict(model).values())), next(iter(raw.values()))))
        for name, value in raw.items():
            torch.testing.assert_close(adapter_state_dict(model)[name], value)
        load_adapter_state_dict(model, state)

    def test_wrapped_layers_are_lora_linear(self):
        model = self._model()
        inject_lora(
            model,
            LoRASection(base_checkpoint="base.pt", preset="proj_lora", rank=4, alpha=4),
        )
        self.assertIsInstance(model.blocks[0].attention.proj, LoRALinear)
        self.assertNotIsInstance(model.blocks[0].attention.qkv, LoRALinear)
        self.assertIsInstance(model.blocks[0].mlp.fc1, LoRALinear)


if __name__ == "__main__":
    unittest.main()
