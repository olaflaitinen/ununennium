"""PINN (Physics-Informed Neural Networks) module."""

from ununennium.models.pinn.base import PINN, MLP, PDEEquation
from ununennium.models.pinn.collocation import (
    AdaptiveSampler,
    CollocationSampler,
    UniformSampler,
)
from ununennium.models.pinn.equations import (
    AdvectionEquation,
    DiffusionEquation,
)

__all__ = [
    "PINN",
    "MLP",
    "AdaptiveSampler",
    "AdvectionEquation",
    "CollocationSampler",
    "DiffusionEquation",
    "PDEEquation",
    "UniformSampler",
]
