"""Tests for PINN models."""

import pytest
import torch

from ununennium.models.pinn import PINN, DiffusionEquation
from ununennium.models.pinn.base import MLP


class TestPINN:
    def test_forward(self):
        equation = DiffusionEquation(diffusivity=0.1)
        network = MLP([2, 32, 1])
        pinn = PINN(network=network, equation=equation)

        x = torch.randn(10, 2, requires_grad=True)
        out = pinn(x)
        assert out.shape == (10, 1)
