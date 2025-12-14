"""Tests for spectral indices."""

import pytest
import torch
from ununennium.preprocessing import ndvi, ndwi, evi

class TestIndices:
    def test_ndvi(self):
        nir = torch.ones(10, 10) * 0.8
        red = torch.ones(10, 10) * 0.2
        # (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
        res = ndvi(nir, red)
        assert torch.allclose(res, torch.tensor(0.6))
        
    def test_ndwi(self):
        green = torch.ones(10, 10) * 0.6
        nir = torch.ones(10, 10) * 0.2
        # (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
        res = ndwi(green, nir)
        assert torch.allclose(res, torch.tensor(0.5))

    def test_evi(self):
        nir = torch.rand(10, 10)
        red = torch.rand(10, 10)
        blue = torch.rand(10, 10)
        res = evi(nir, red, blue)
        assert res.shape == (10, 10)
