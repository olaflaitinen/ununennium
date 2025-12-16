"""Losses module for training."""

from ununennium.losses.detection import (
    DetectionLoss,
    FocalLossDetection,
    GIoULoss,
    SmoothL1Loss,
)
from ununennium.losses.segmentation import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
)

__all__ = [
    "CombinedLoss",
    "DetectionLoss",
    "DiceLoss",
    "FocalLoss",
    "FocalLossDetection",
    "GIoULoss",
    "SmoothL1Loss",
]
