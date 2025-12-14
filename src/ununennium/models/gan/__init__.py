"""GAN module for image-to-image translation."""

from ununennium.models.gan.generators import (
    UNetGenerator,
    ResNetGenerator,
)
from ununennium.models.gan.discriminators import (
    PatchDiscriminator,
    MultiScaleDiscriminator,
)
from ununennium.models.gan.losses import (
    AdversarialLoss,
    PerceptualLoss,
    SpectralAngleLoss,
)
from ununennium.models.gan.pix2pix import Pix2Pix
from ununennium.models.gan.cyclegan import CycleGAN

__all__ = [
    "UNetGenerator",
    "ResNetGenerator",
    "PatchDiscriminator",
    "MultiScaleDiscriminator",
    "AdversarialLoss",
    "PerceptualLoss",
    "SpectralAngleLoss",
    "Pix2Pix",
    "CycleGAN",
]
