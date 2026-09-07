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

    def test_adapter_ema_load_preserves_shadow_dtype(self):
        model = self._model().to(dtype=torch.float64)
        inject_lora(
            model,
            LoRASection(base_checkpoint="base.pt", preset="mlp_lora", rank=4, alpha=4),
        )
        ema = AdapterEMA(model, 0.9)
        checkpoint_state = ema.state_dict()
        checkpoint_state["adapter"] = {
            name: value.float() for name, value in checkpoint_state["adapter"].items()
        }
        ema.load_state_dict(checkpoint_state)
        self.assertTrue(all(value.dtype == torch.float64 for value in ema.shadow.values()))

    def test_wrapped_layers_are_lora_linear(self):
        model = self._model()
        inject_lora(
            model,
            LoRASection(base_checkpoint="base.pt", preset="proj_lora", rank=4, alpha=4),
        )
        self.assertIsInstance(model.blocks[0].attention.proj, LoRALinear)
        self.assertNotIsInstance(model.blocks[0].attention.qkv, LoRALinear)
        self.assertIsInstance(model.blocks[0].mlp.fc1, LoRALinear)

    def test_adapters_inherit_wrapped_layer_device_and_dtype(self):
        meta_wrapper = LoRALinear(
            torch.nn.Linear(4, 4, device="meta"), rank=2, alpha=2, dropout=0.0
        )
        self.assertEqual(meta_wrapper.lora_a.device, torch.device("meta"))
        self.assertEqual(meta_wrapper.lora_b.device, torch.device("meta"))

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                model = self._model().to(device=device, dtype=torch.float64)
                inject_lora(
                    model,
                    LoRASection(
                        base_checkpoint="base.pt", preset="full_lora",
                        rank=4, alpha=4,
                    ),
                )
                for parameter in adapter_parameters(model):
                    self.assertEqual(parameter.device, device)
                    self.assertEqual(parameter.dtype, torch.float64)
                output = model(
                    torch.randn(2, 2, 4, 4, device=device, dtype=torch.float64),
                    torch.rand(2, device=device),
                )
                self.assertEqual(output.device, device)
                self.assertEqual(output.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
