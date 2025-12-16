"""PINN (Physics-Informed Neural Networks) module."""

from ununennium.models.pinn.base import MLP, PDEEquation, PINN

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
    "AdaptiveSampler",
    "AdvectionEquation",
    "CollocationSampler",
    "DiffusionEquation",
    "MLP",
    "PDEEquation",
    "PINN",
    "UniformSampler",
]
