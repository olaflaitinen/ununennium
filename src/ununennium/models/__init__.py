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
    "FCOS",
    "PINN",
    "ClassificationHead",
    "CycleGAN",
    "DetectionHead",
    "EfficientNetBackbone",
    "FasterRCNN",
    "Pix2Pix",
    "ResNetBackbone",
    "RetinaNet",
    "SegmentationHead",
    "UNet",
    "create_model",
    "list_models",
    "register_model",
]
