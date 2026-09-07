"""One-update cluster smoke test for a configured DiT."""

from pathlib import Path

import torch

from dragn.config import ExperimentConfig
from dragn.models import build_dit


def run_dit_smoke(config: ExperimentConfig) -> Path:
    if config.task != "base" or config.model is None:
        raise ValueError("DiT smoke test requires task: base and a model section")
    torch.manual_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (
        config.data.image_size // config.autoencoder.downsample_factor
        if config.autoencoder.enabled
        else config.data.image_size
    )
    in_channels = config.autoencoder.latent_channels if config.autoencoder.enabled else config.data.channels
    model = build_dit(config.model, in_channels, input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=0.0)
    inputs = torch.randn(1, in_channels, input_size, input_size, device=device)
    time = torch.tensor([0.5], device=device)
    target = torch.randn_like(inputs)

    model.train()
    initial = model(inputs, time)
    if torch.count_nonzero(initial).item() != 0:
        raise RuntimeError("AdaLN-Zero DiT must produce exact zeros before its first update")
    loss = torch.nn.functional.mse_loss(initial, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = model.final_layer.projection.weight.grad.norm().item()
    if not torch.isfinite(model.final_layer.projection.weight.grad).all() or gradient_norm == 0:
        raise RuntimeError("DiT final projection did not receive a finite nonzero gradient")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        updated = model(inputs, time)
    if torch.count_nonzero(updated).item() == 0:
        raise RuntimeError("DiT output remained zero after an optimizer update")

    output_dir = config.experiment.output_dir / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_smoke.pt"
    torch.save({
        "schema_version": 2,
        "task": "base",
        "model_state": model.state_dict(),
        "config": config.model_dump(mode="json"),
    }, checkpoint_path)
    reloaded = build_dit(config.model, in_channels, input_size).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_output = reloaded(inputs, time)
    if not torch.equal(updated, reloaded_output):
        raise RuntimeError("Reloaded DiT output does not exactly match the saved model output")

    print(f"device={device}")
    print(f"input_shape={tuple(inputs.shape)} output_shape={tuple(updated.shape)}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"loss={loss.item():.6f} final_projection_gradient_norm={gradient_norm:.6f}")
    print(f"checkpoint_reload=exact")
    print(f"checkpoint={checkpoint_path}")
    return checkpoint_path
