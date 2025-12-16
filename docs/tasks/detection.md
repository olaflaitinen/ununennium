# Object Detection

Localize objects in satellite imagery with bounding box predictions.

---

## Overview

Object detection identifies and localizes objects of interest (vehicles, buildings, ships, aircraft, etc.) in geospatial imagery by predicting bounding boxes and class labels.

### Mathematical Formulation

For each detection, the model predicts:

- **Classification**: $P(c_i | x) \in [0, 1]^K$ for $K$ classes
- **Bounding Box**: $(x, y, w, h)$ or $(x_1, y_1, x_2, y_2)$ coordinates
- **Confidence**: Objectness score $P(\text{object})$

The detection loss combines classification and localization:

$$\mathcal{L} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{reg} \mathcal{L}_{reg}$$

---

## Available Architectures

| Model | Type | Anchors | Use Case |
|-------|------|---------|----------|
| **RetinaNet** | One-stage | Dense | General detection, good accuracy |
| **Faster R-CNN** | Two-stage | RPN | High accuracy, larger objects |
| **FCOS** | Anchor-free | None | Simple, efficient, small objects |

---

## Quick Start

### Basic Detection

```python
import torch
from ununennium.models import create_model

# Create RetinaNet for satellite imagery (12 bands)
model = create_model(
    "retinanet",
    in_channels=12,      # Sentinel-2 bands
    num_classes=5,       # Object classes
    backbone="resnet50",
)

# Forward pass
x = torch.randn(1, 12, 512, 512)
output = model(x)

# Access predictions
for level_cls, level_box in zip(output.class_logits, output.box_regression):
    print(f"Level shape: cls={level_cls.shape}, box={level_box.shape}")
```

### Training with Detection Loss

```python
from ununennium.losses import DetectionLoss, FocalLossDetection, GIoULoss

# Combined loss for object detection
loss_fn = DetectionLoss(
    cls_loss=FocalLossDetection(alpha=0.25, gamma=2.0),
    reg_loss=GIoULoss(),
    cls_weight=1.0,
    reg_weight=1.0,
)

# Compute loss
losses = loss_fn(cls_pred, cls_target, reg_pred, reg_target)
print(f"Total: {losses['total_loss']:.4f}")
```

### Evaluation with mAP

```python
from ununennium.metrics import mean_average_precision

# Compute COCO-style mAP
results = mean_average_precision(
    pred_boxes=pred_boxes,
    pred_scores=pred_scores,
    pred_labels=pred_labels,
    gt_boxes=gt_boxes,
    gt_labels=gt_labels,
)

print(f"mAP: {results['mAP']:.3f}")
print(f"AP50: {results['AP50']:.3f}")
print(f"AP75: {results['AP75']:.3f}")
```

---

## Loss Functions

| Loss | Description | Use Case |
|------|-------------|----------|
| `FocalLossDetection` | Down-weights easy negatives | Classification with class imbalance |
| `SmoothL1Loss` | Robust regression loss | Bounding box regression |
| `GIoULoss` | Generalized IoU loss | Better box alignment |
| `DetectionLoss` | Combined cls + reg | End-to-end training |

---

## Metrics

| Metric | Description |
|--------|-------------|
| `mean_average_precision` | COCO-style mAP@[0.5:0.95] |
| `average_precision_at_iou` | AP at specific IoU threshold |
| `compute_iou_boxes` | Pairwise box IoU |

---

## Remote Sensing Applications

- **Vehicle Detection**: Cars, trucks, aircraft on runways
- **Ship Detection**: Maritime surveillance with SAR/optical
- **Building Detection**: Urban mapping and change detection
- **Solar Panel Detection**: Renewable energy infrastructure
- **Pool Detection**: Urban planning and water management

---

## Hyperparameter Guidance

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| FPN channels | 256 | Standard for most backbones |
| Anchor scales | (32, 64, 128, 256, 512) | Match object sizes in dataset |
| Aspect ratios | (0.5, 1.0, 2.0) | Cover common object shapes |
| Focal γ | 2.0 | Higher = more focus on hard examples |
| Focal α | 0.25 | Balance positive/negative |
