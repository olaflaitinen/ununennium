"""Models module with architectures for remote sensing tasks."""

from ununennium.models.architectures.unet import UNet
from ununennium.models.backbones import (
    EfficientNetBackbone,
    ResNetBackbone,
)
from ununennium.models.gan import CycleGAN, Pix2Pix
from ununennium.models.heads import (
    ClassificationHead,
    SegmentationHead,
)
from ununennium.models.pinn import PINN
from ununennium.models.registry import create_model, list_models, register_model

__all__ = [
    "PINN",
    "ClassificationHead",
    "CycleGAN",
    "EfficientNetBackbone",
    "Pix2Pix",
    "ResNetBackbone",
    "SegmentationHead",
    "UNet",
    "create_model",
    "list_models",
    "register_model",
]
