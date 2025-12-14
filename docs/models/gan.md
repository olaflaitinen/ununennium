# GAN Models

Generative Adversarial Networks for image translation.

## Pix2Pix

Paired image-to-image translation.

```python
from ununennium.models.gan import Pix2Pix

model = Pix2Pix(in_channels=12, out_channels=3)
fake_b = model(real_a)
```

## CycleGAN

Unpaired image-to-image translation.

```python
from ununennium.models.gan import CycleGAN

model = CycleGAN(in_channels_a=2, in_channels_b=3)
fake_optical = model(sar_image, direction="A2B")
```
