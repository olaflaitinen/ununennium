"""Tests for GAN models."""

import pytest
import torch

from ununennium.models.gan import Pix2Pix, CycleGAN


class TestPix2Pix:
    def test_forward(self):
        model = Pix2Pix(in_channels=12, out_channels=3)
        x = torch.randn(2, 12, 256, 256)
        out = model(x)
        assert out.shape == (2, 3, 256, 256)


class TestCycleGAN:
    def test_forward(self):
        model = CycleGAN(in_channels_a=2, in_channels_b=3)
        x = torch.randn(2, 2, 128, 128)
        out = model(x, direction="A2B")
        assert out.shape == (2, 3, 128, 128)
