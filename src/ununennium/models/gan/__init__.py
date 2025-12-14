"""GAN module for image-to-image translation."""

from ununennium.models.gan.cyclegan import CycleGAN
from ununennium.models.gan.discriminators import (
    MultiScaleDiscriminator,
    PatchDiscriminator,
)
from ununennium.models.gan.generators import (
    ResNetGenerator,
    UNetGenerator,
)
from ununennium.models.gan.losses import (
    AdversarialLoss,
    PerceptualLoss,
    SpectralAngleLoss,
)
from ununennium.models.gan.pix2pix import Pix2Pix

__all__ = [
    "AdversarialLoss",
    "CycleGAN",
    "MultiScaleDiscriminator",
    "PatchDiscriminator",
    "PerceptualLoss",
    "Pix2Pix",
    "ResNetGenerator",
    "SpectralAngleLoss",
    "UNetGenerator",
]
