"""Models module with architectures for remote sensing tasks."""

from ununennium.models.architectures.detection import (
    FCOS,
    FasterRCNN,
    RetinaNet,
)
from ununennium.models.architectures.unet import UNet
from ununennium.models.backbones import (
    EfficientNetBackbone,
    ResNetBackbone,
)
from ununennium.models.gan import CycleGAN, Pix2Pix
from ununennium.models.heads import (
    ClassificationHead,
    DetectionHead,
    SegmentationHead,
)
from ununennium.models.pinn import PINN
from ununennium.models.registry import create_model, list_models, register_model

__all__ = [
    "ClassificationHead",
    "create_model",
    "CycleGAN",
    "DetectionHead",
    "EfficientNetBackbone",
    "FasterRCNN",
    "FCOS",
    "list_models",
    "PINN",
    "Pix2Pix",
    "register_model",
    "ResNetBackbone",
    "RetinaNet",
    "SegmentationHead",
    "UNet",
]

