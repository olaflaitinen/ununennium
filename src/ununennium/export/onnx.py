"""ONNX model export."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...],
    opset_version: int = 17,
) -> Path:
    """Export model to ONNX format.

    Args:
        model: PyTorch model.
        output_path: Output file path.
        input_shape: Example input shape.
        opset_version: ONNX opset version.

    Returns:
        Path to exported model.
    """
    output_path = Path(output_path)
    model.eval()

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    return output_path
