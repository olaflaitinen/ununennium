"""PINN (Physics-Informed Neural Networks) module."""

from ununennium.models.pinn.base import MLP, PINN, PDEEquation
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
    "MLP",
    "PINN",
    "AdaptiveSampler",
    "AdvectionEquation",
    "CollocationSampler",
    "DiffusionEquation",
    "PDEEquation",
    "UniformSampler",
]
