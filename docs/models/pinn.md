# Physics-Informed Neural Networks

PDE-constrained learning for physical consistency.

## Usage

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP

equation = DiffusionEquation(diffusivity=0.1)
network = MLP([2, 64, 64, 1])
pinn = PINN(network=network, equation=equation)

loss = pinn.compute_loss(x_data, u_data, x_collocation)
```

## Available Equations

- DiffusionEquation
- AdvectionEquation
- AdvectionDiffusionEquation
