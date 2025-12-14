"""Tests for model layers."""

import pytest
import torch
from ununennium.models.layers import DoubleConv, Down, Up

class TestLayers:
    def test_double_conv(self):
        layer = DoubleConv(3, 64)
        x = torch.randn(1, 3, 128, 128)
        out = layer(x)
        assert out.shape == (1, 64, 128, 128)

    def test_down(self):
        layer = Down(64, 128)
        x = torch.randn(1, 64, 128, 128)
        out = layer(x)
        assert out.shape == (1, 128, 64, 64)

    def test_up(self):
        layer = Up(128, 64)
        x1 = torch.randn(1, 128, 32, 32) # From deeper layer
        x2 = torch.randn(1, 64, 64, 64)  # Skip connection
        out = layer(x1, x2)
        assert out.shape == (1, 64, 64, 64)
