"""Pytest configuration and fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Create a sample 12-band image tensor."""
    return torch.randn(12, 256, 256)


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Create a batch of sample images."""
    return torch.randn(4, 12, 256, 256)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Create sample segmentation labels."""
    return torch.randint(0, 10, (4, 256, 256))


@pytest.fixture
def sample_crs_string() -> str:
    """Sample CRS as EPSG code string."""
    return "EPSG:32632"  # UTM zone 32N


@pytest.fixture
def sample_transform():
    """Sample affine transform."""
    from affine import Affine

    return Affine(10, 0, 500000, 0, -10, 5000000)


@pytest.fixture
def device() -> str:
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"
