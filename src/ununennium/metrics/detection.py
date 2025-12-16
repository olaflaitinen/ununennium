"""Detection metrics for object detection evaluation.

Implements standard object detection metrics:
- mean_average_precision: mAP across IoU thresholds
- average_precision_at_iou: AP at specific IoU threshold
- compute_iou_boxes: IoU computation for bounding boxes
"""

from __future__ import annotations

import torch


def compute_iou_boxes(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: First set of boxes (N, 4) in (x1, y1, x2, y2) format.
        boxes2: Second set of boxes (M, 4) in (x1, y1, x2, y2) format.

    Returns:
        IoU matrix of shape (N, M).
    """
    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    union_area = area1[:, None] + area2 - inter_area

    return inter_area / (union_area + 1e-7)


def average_precision_at_iou(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute Average Precision at a specific IoU threshold.

    Args:
        pred_boxes: Predicted boxes (N, 4).
        pred_scores: Prediction confidence scores (N,).
        pred_labels: Predicted class labels (N,).
        gt_boxes: Ground truth boxes (M, 4).
        gt_labels: Ground truth class labels (M,).
        iou_threshold: IoU threshold for matching.
        num_classes: Number of classes (auto-detected if None).

    Returns:
        Dictionary with per-class AP and mean AP.
    """
    if num_classes is None:
        all_labels = torch.cat([pred_labels, gt_labels])
        num_classes = int(all_labels.max().item()) + 1 if len(all_labels) > 0 else 1

    ap_per_class = {}

    for cls in range(num_classes):
        # Filter by class
        cls_mask_pred = pred_labels == cls
        cls_mask_gt = gt_labels == cls

        cls_pred_boxes = pred_boxes[cls_mask_pred]
        cls_pred_scores = pred_scores[cls_mask_pred]
        cls_gt_boxes = gt_boxes[cls_mask_gt]

        n_gt = cls_gt_boxes.shape[0]
        n_pred = cls_pred_boxes.shape[0]

        if n_gt == 0:
            ap_per_class[f"class_{cls}"] = 0.0 if n_pred > 0 else 1.0
            continue

        if n_pred == 0:
            ap_per_class[f"class_{cls}"] = 0.0
            continue

        # Sort by confidence
        sorted_indices = torch.argsort(cls_pred_scores, descending=True)
        cls_pred_boxes = cls_pred_boxes[sorted_indices]

        # Compute IoU
        ious = compute_iou_boxes(cls_pred_boxes, cls_gt_boxes)

        # Match predictions to ground truth
        gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=gt_boxes.device)
        tp = torch.zeros(n_pred, device=pred_boxes.device)
        fp = torch.zeros(n_pred, device=pred_boxes.device)

        for i in range(n_pred):
            max_iou, max_idx = ious[i].max(dim=0)
            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                tp[i] = 1
                gt_matched[max_idx] = True
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        recall = tp_cumsum / n_gt

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max().item() / 11

        ap_per_class[f"class_{cls}"] = ap

    # Mean AP
    ap_values = list(ap_per_class.values())
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else 0.0

    return {"mAP": mean_ap, **ap_per_class}


def mean_average_precision(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresholds: list[float] | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute mean Average Precision across multiple IoU thresholds.

    Computes mAP in COCO-style: averaged over IoU thresholds [0.5:0.95:0.05].

    Args:
        pred_boxes: Predicted boxes (N, 4).
        pred_scores: Prediction confidence scores (N,).
        pred_labels: Predicted class labels (N,).
        gt_boxes: Ground truth boxes (M, 4).
        gt_labels: Ground truth class labels (M,).
        iou_thresholds: List of IoU thresholds (default: [0.5:0.95:0.05]).
        num_classes: Number of classes.

    Returns:
        Dictionary with mAP, AP50, AP75, and per-threshold APs.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    results = {}
    all_maps = []

    for iou_thresh in iou_thresholds:
        ap_result = average_precision_at_iou(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=iou_thresh,
            num_classes=num_classes,
        )
        results[f"mAP@{iou_thresh:.2f}"] = ap_result["mAP"]
        all_maps.append(ap_result["mAP"])

    # Standard metrics
    results["mAP"] = sum(all_maps) / len(all_maps) if all_maps else 0.0
    results["AP50"] = results.get("mAP@0.50", 0.0)
    results["AP75"] = results.get("mAP@0.75", 0.0)

    return results
