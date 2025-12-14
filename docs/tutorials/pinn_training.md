# PINN Training Tutorial

Train Physics-Informed Neural Networks.

## Setup

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP, UniformSampler
import torch

# Define equation
equation = DiffusionEquation(diffusivity=0.1)

# Create network
network = MLP([2, 64, 64, 64, 1], activation="tanh")

# Create PINN
pinn = PINN(
    network=network,
    equation=equation,
    lambda_data=1.0,
    lambda_pde=10.0,
)

# Collocation sampler
sampler = UniformSampler(bounds=[(0, 1), (0, 1)])

# Training
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

for epoch in range(epochs):
    x_collocation = sampler.sample(1000)

    losses = pinn.compute_loss(
        x_data=x_data,
        u_data=u_data,
        x_collocation=x_collocation,
    )

    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
```
