"""TorchScript model export."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...],
    method: str = "trace",
) -> Path:
    """Export model to TorchScript format.

    Args:
        model: PyTorch model.
        output_path: Output file path.
        input_shape: Example input shape.
        method: "trace" or "script".

    Returns:
        Path to exported model.
    """
    output_path = Path(output_path)
    model.eval()

    dummy_input = torch.randn(*input_shape)

    scripted = torch.jit.trace(model, dummy_input) if method == "trace" else torch.jit.script(model)

    scripted.save(str(output_path))

    return output_path
