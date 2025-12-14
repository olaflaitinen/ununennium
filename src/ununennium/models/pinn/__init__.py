"""PINN (Physics-Informed Neural Networks) module."""

from ununennium.models.pinn.base import PDEEquation, PINN
from ununennium.models.pinn.equations import (
    DiffusionEquation,
    AdvectionEquation,
)
from ununennium.models.pinn.collocation import (
    CollocationSampler,
    UniformSampler,
    AdaptiveSampler,
)

__all__ = [
    "PDEEquation",
    "PINN",
    "DiffusionEquation",
    "AdvectionEquation",
    "CollocationSampler",
    "UniformSampler",
    "AdaptiveSampler",
]
