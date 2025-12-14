"""Segmentation metrics."""

from __future__ import annotations

import torch


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute Intersection over Union (IoU) per class.

    Args:
        pred: Predicted labels (B, H, W) or logits (B, C, H, W).
        target: Ground truth labels (B, H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore in computation.

    Returns:
        IoU score per class (num_classes,).
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    iou_per_class = torch.zeros(num_classes, device=pred.device)

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        mask = target != ignore_index

        intersection = (pred_c & target_c & mask).sum().float()
        union = ((pred_c | target_c) & mask).sum().float()

        if union > 0:
            iou_per_class[c] = intersection / union

    return iou_per_class


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice score per class.

    Args:
        pred: Predicted labels or logits.
        target: Ground truth labels.
        num_classes: Number of classes.
        smooth: Smoothing factor.

    Returns:
        Dice score per class.
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    dice_per_class = torch.zeros(num_classes, device=pred.device)

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        dice_per_class[c] = (2 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)

    return dice_per_class


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute overall pixel accuracy.

    Args:
        pred: Predicted labels or logits.
        target: Ground truth labels.

    Returns:
        Pixel accuracy as float.
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    correct = (pred == target).sum().float()
    total = target.numel()

    return (correct / total).item()
