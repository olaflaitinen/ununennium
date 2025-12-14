"""GAN training examples."""

from ununennium.models.gan import Pix2Pix
import torch

model = Pix2Pix(in_channels=12, out_channels=3)
print("Pix2Pix model created")
