"""Detection losses for object detection training.

Implements loss functions for bounding box detection:
- FocalLossDetection: Class imbalance-aware classification loss
- SmoothL1Loss: Robust bounding box regression loss
- GIoULoss: Generalized IoU loss for better box regression
- DetectionLoss: Combined classification and regression loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLossDetection(nn.Module):
    """Focal Loss for dense object detection.

    Addresses class imbalance between foreground and background
    by down-weighting easy negatives.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive examples.
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: Reduction method ('none', 'mean', 'sum').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits of shape (N, C) or (N, C, H, W).
            target: Target labels of shape (N,) or (N, H, W).

        Returns:
            Focal loss value.
        """
        # Flatten spatial dimensions if present
        if pred.dim() == 4:
            pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
            target = target.reshape(-1)

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction="none")

        # Compute focal weight
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(target > 0, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) Loss for bounding box regression.

    Combines the best properties of L1 and L2 losses:
    - L2 for small errors (smooth gradient near zero)
    - L1 for large errors (robust to outliers)

    Loss = 0.5 * x^2 / β       if |x| < β
           |x| - 0.5 * β       otherwise
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize Smooth L1 Loss.

        Args:
            beta: Threshold for switching between L1 and L2.
            reduction: Reduction method ('none', 'mean', 'sum').
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute smooth L1 loss.

        Args:
            pred: Predicted boxes of shape (N, 4) or (N, A*4, H, W).
            target: Target boxes of same shape as pred.
            weights: Optional per-element weights.

        Returns:
            Smooth L1 loss value.
        """
        diff = pred - target
        abs_diff = torch.abs(diff)

        loss = torch.where(
            abs_diff < self.beta,
            0.5 * diff**2 / self.beta,
            abs_diff - 0.5 * self.beta,
        )

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GIoULoss(nn.Module):
    """Generalized Intersection over Union Loss.

    Improves upon IoU loss by also considering the smallest
    enclosing box, providing gradients even when boxes don't overlap.

    GIoU = IoU - (C - U) / C

    where C is the smallest enclosing box area and U is the union area.

    Reference:
        Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize GIoU Loss.

        Args:
            reduction: Reduction method ('none', 'mean', 'sum').
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GIoU loss.

        Args:
            pred: Predicted boxes of shape (N, 4) in (x1, y1, x2, y2) format.
            target: Target boxes of shape (N, 4).

        Returns:
            GIoU loss value (1 - GIoU).
        """
        # Ensure valid box coordinates
        pred_x1 = torch.min(pred[:, 0], pred[:, 2])
        pred_y1 = torch.min(pred[:, 1], pred[:, 3])
        pred_x2 = torch.max(pred[:, 0], pred[:, 2])
        pred_y2 = torch.max(pred[:, 1], pred[:, 3])

        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 2]
        target_y2 = target[:, 3]

        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        # Areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-7)

        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        loss = 1 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DetectionLoss(nn.Module):
    """Combined loss for object detection.

    Combines classification and regression losses with configurable weights.

    Total Loss = λ_cls * L_cls + λ_reg * L_reg
    """

    def __init__(
        self,
        cls_loss: nn.Module | None = None,
        reg_loss: nn.Module | None = None,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
    ):
        """Initialize detection loss.

        Args:
            cls_loss: Classification loss (default: FocalLossDetection).
            reg_loss: Regression loss (default: SmoothL1Loss).
            cls_weight: Weight for classification loss.
            reg_weight: Weight for regression loss.
        """
        super().__init__()
        self.cls_loss = cls_loss or FocalLossDetection()
        self.reg_loss = reg_loss or SmoothL1Loss()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_pred: torch.Tensor,
        reg_target: torch.Tensor,
        reg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined detection loss.

        Args:
            cls_pred: Classification predictions.
            cls_target: Classification targets.
            reg_pred: Regression predictions.
            reg_target: Regression targets.
            reg_mask: Optional mask for valid regression samples.

        Returns:
            Dictionary with 'cls_loss', 'reg_loss', and 'total_loss'.
        """
        cls_loss = self.cls_loss(cls_pred, cls_target)

        if reg_mask is not None:
            reg_pred = reg_pred[reg_mask]
            reg_target = reg_target[reg_mask]

        reg_loss = self.reg_loss(reg_pred, reg_target)

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }
