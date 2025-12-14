"""PINN training example."""

from ununennium.models.pinn import PINN, DiffusionEquation
from ununennium.models.pinn.base import MLP

equation = DiffusionEquation(diffusivity=0.1)
network = MLP([2, 64, 64, 1])
pinn = PINN(network=network, equation=equation)
print("PINN model created")
