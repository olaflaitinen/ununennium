"""Visualization utilities for geospatial data."""

import math

import matplotlib.pyplot as plt  # type: ignore
import torch
from matplotlib.figure import Figure  # type: ignore

from ununennium.core import GeoTensor
from ununennium.core.band_specs import get_rgb_bands


def plot_rgb(
    tensor: GeoTensor,
    sensor: str | None = None,
    bands: tuple[str, str, str] | None = None,
    brightness: float = 1.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot an RGB composite from a GeoTensor.

    Args:
        tensor: Input GeoTensor.
        sensor: Sensor name to automatically select RGB bands.
        bands: Explicit list of 3 band names for RGB.
        brightness: Brightness factor.
        ax: Matplotlib axes.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10))

    if bands is None:
        if sensor:
            bands = get_rgb_bands(sensor)
        else:
            # Default to first 3 bands
            bands = tuple(tensor.band_names[:3]) if tensor.band_names and len(tensor.band_names) >= 3 else None  # type: ignore

    if bands is None or len(bands) != 3:
        raise ValueError("Must provide 3 bands for RGB plotting.")

    # Select indices
    try:
        indices = [tensor.band_names.index(b) for b in bands]
    except ValueError as e:
        raise ValueError(f"Band not found in tensor: {e}") from e

    rgb = tensor.data[indices, :, :].float()

    # Normalize
    p2 = torch.quantile(rgb, 0.02)
    p98 = torch.quantile(rgb, 0.98)
    rgb = (rgb - p2) / (p98 - p2)
    rgb = torch.clamp(rgb * brightness, 0, 1)

    # To channel last numpy
    rgb_np = rgb.permute(1, 2, 0).cpu().numpy()

    ax.imshow(rgb_np)
    ax.set_title(f"RGB Composite ({', '.join(bands)})")
    ax.axis("off")
    return ax


def plot_bands(
    tensor: GeoTensor,
    bands: list[str] | None = None,
    cols: int = 4,
    cmap: str = "viridis",
) -> Figure:
    """Plot individual bands in a grid.

    Args:
        tensor: Input GeoTensor.
        bands: List of bands to plot.
        cols: Number of columns.
        cmap: Colormap.

    Returns:
        Matplotlib figure.
    """
    if bands is None:
        bands = tensor.band_names

    n = len(bands)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, band in enumerate(bands):
        idx = tensor.band_names.index(band)
        data = tensor.data[idx].float().cpu().numpy()

        ax = axes[i]
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(band)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide empty axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig
