"""Segmentation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-6, multiclass: bool = True):
        super().__init__()
        self.smooth = smooth
        self.multiclass = multiclass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.multiclass:
            pred = F.softmax(pred, dim=1)
            num_classes = pred.shape[1]

            target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

            intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
            union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()
        else:
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combine multiple losses with weights."""

    def __init__(self, losses: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device)
        for loss, weight in zip(self.losses, self.weights, strict=False):
            total = total + weight * loss(pred, target)
        return total
