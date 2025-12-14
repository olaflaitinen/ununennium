"""CycleGAN training example."""

from ununennium.models.gan import CycleGAN
import torch

model = CycleGAN(in_channels_a=2, in_channels_b=3)
print("CycleGAN model created")
