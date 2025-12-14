"""Normalization utilities for satellite imagery."""

from __future__ import annotations

from typing import Literal

import torch


def normalize(
    data: torch.Tensor,
    method: Literal["minmax", "zscore", "percentile"] = "minmax",
    per_channel: bool = True,
    percentiles: tuple[float, float] = (2, 98),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize tensor data.

    Args:
        data: Input tensor of shape (..., C, H, W) or (C, H, W).
        method: Normalization method to use.
        per_channel: Whether to normalize each channel independently.
        percentiles: Percentiles for percentile normalization.

    Returns:
        Tuple of (normalized_data, stats_dict) where stats_dict contains
        the statistics needed for denormalization.

    Example:
        >>> normalized, stats = normalize(data, method="zscore")
        >>> original = denormalize(normalized, stats, method="zscore")
    """
    if method == "minmax":
        if per_channel:
            # Shape: (C,) for each stat
            dims = (*tuple(range(data.ndim - 2)), -2, -1)
            min_vals = data.amin(dim=dims, keepdim=True)
            max_vals = data.amax(dim=dims, keepdim=True)
        else:
            min_vals = data.min()
            max_vals = data.max()

        denominator = max_vals - min_vals
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        normalized = (data - min_vals) / denominator
        stats = {"min": min_vals, "max": max_vals}

    elif method == "zscore":
        if per_channel:
            dims = (*tuple(range(data.ndim - 2)), -2, -1)
            mean = data.mean(dim=dims, keepdim=True)
            std = data.std(dim=dims, keepdim=True)
        else:
            mean = data.mean()
            std = data.std()

        std = torch.where(std == 0, torch.ones_like(std), std)
        normalized = (data - mean) / std
        stats = {"mean": mean, "std": std}

    elif method == "percentile":
        if per_channel:
            # Per-channel percentile normalization
            flat = data.flatten(-2, -1)
            low = torch.quantile(flat, percentiles[0] / 100, dim=-1, keepdim=True)
            high = torch.quantile(flat, percentiles[1] / 100, dim=-1, keepdim=True)
            low = low.unsqueeze(-1)
            high = high.unsqueeze(-1)
        else:
            low = torch.quantile(data, percentiles[0] / 100)
            high = torch.quantile(data, percentiles[1] / 100)

        denominator = high - low
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        normalized = (data - low) / denominator
        normalized = torch.clamp(normalized, 0, 1)
        stats = {"low": low, "high": high}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, stats


def denormalize(
    data: torch.Tensor,
    stats: dict[str, torch.Tensor],
    method: Literal["minmax", "zscore", "percentile"] = "minmax",
) -> torch.Tensor:
    """Reverse normalization using saved statistics.

    Args:
        data: Normalized tensor.
        stats: Statistics dictionary from normalize().
        method: Normalization method that was used.

    Returns:
        Denormalized tensor.
    """
    if method == "minmax":
        min_vals = stats["min"]
        max_vals = stats["max"]
        return data * (max_vals - min_vals) + min_vals

    elif method == "zscore":
        mean = stats["mean"]
        std = stats["std"]
        return data * std + mean

    elif method == "percentile":
        low = stats["low"]
        high = stats["high"]
        return data * (high - low) + low

    else:
        raise ValueError(f"Unknown normalization method: {method}")
