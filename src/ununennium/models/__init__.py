"""Models module with architectures for remote sensing tasks."""

from ununennium.models.registry import create_model, register_model, list_models
from ununennium.models.backbones import (
    ResNetBackbone,
    EfficientNetBackbone,
)
from ununennium.models.heads import (
    ClassificationHead,
    SegmentationHead,
)
from ununennium.models.architectures.unet import UNet
from ununennium.models.gan import Pix2Pix, CycleGAN
from ununennium.models.pinn import PINN

__all__ = [
    "create_model",
    "register_model",
    "list_models",
    "ResNetBackbone",
    "EfficientNetBackbone",
    "ClassificationHead",
    "SegmentationHead",
    "UNet",
    "Pix2Pix",
    "CycleGAN",
    "PINN",
]
