"""Tests for preprocessing."""

import pytest
import torch
from ununennium.preprocessing import normalize, denormalize

class TestNormalization:
    def test_minmax(self):
        # Shape (C, H, W) -> (1, 2, 2)
        x = torch.tensor([[[0.0, 50.0], [50.0, 100.0]]])
        # Returns tuple (normalized, stats)
        norm, stats = normalize(x, method="minmax")
        expected = torch.tensor([[[0.0, 0.5], [0.5, 1.0]]])
        assert torch.allclose(norm, expected)
        
        denorm = denormalize(norm, stats, method="minmax")
        assert torch.allclose(denorm, x)

    def test_zscore(self):
        # Shape (1, 10, 10)
        x = torch.randn(1, 10, 10)
        norm, stats = normalize(x, method="zscore")
        assert torch.abs(norm.mean()) < 1e-5
        assert torch.abs(norm.std() - 1.0) < 1e-5
        
    def test_percentile(self):
        # Shape (1, 10, 10)
        x = torch.rand(1, 10, 10)
        norm, stats = normalize(x, method="percentile", percentiles=(0, 100))
        assert norm.min() >= 0
        assert norm.max() <= 1
