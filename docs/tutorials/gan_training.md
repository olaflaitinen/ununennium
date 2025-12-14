# GAN Training Tutorial

Train GANs for image translation.

## Pix2Pix Training

```python
from ununennium.models.gan import Pix2Pix
import torch

model = Pix2Pix(in_channels=12, out_channels=3)
opt_g = torch.optim.Adam(model.generator.parameters(), lr=2e-4)
opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4)

for epoch in range(epochs):
    for real_a, real_b in dataloader:
        # Train generator
        opt_g.zero_grad()
        g_losses = model.compute_generator_loss(real_a, real_b)
        g_losses["total"].backward()
        opt_g.step()

        # Train discriminator
        opt_d.zero_grad()
        d_losses = model.compute_discriminator_loss(
            real_a, real_b, g_losses["fake_b"].detach()
        )
        d_losses["total"].backward()
        opt_d.step()
```
