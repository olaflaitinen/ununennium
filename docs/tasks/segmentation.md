# Semantic Segmentation

Pixel-wise classification for land cover and land use mapping.

## Task Definition

Given an input image `X ∈ ℝ^(C×H×W)`, predict a label map `Y ∈ {1,...,K}^(H×W)`.

### Mathematical Formulation

```
f_θ: ℝ^(C×H×W) → ℝ^(K×H×W)
Y = argmax_k f_θ(X)_k
```

## Common Architectures

### Encoder-Decoder Networks

| Architecture | Parameters | mIoU (LandCover) | Speed |
|--------------|------------|------------------|-------|
| U-Net | 14M - 32M | 0.72 - 0.78 | Fast |
| DeepLabV3+ | 26M - 43M | 0.74 - 0.80 | Moderate |
| FPN | 28M - 45M | 0.73 - 0.79 | Moderate |
| HRNet | 65M - 78M | 0.77 - 0.82 | Slow |

### Skip Connection Types

| Type | Description | Use Case |
|------|-------------|----------|
| Concatenation | `[F_enc, F_dec]` | U-Net |
| Summation | `F_enc + F_dec` | ResNet-style |
| Attention | `α × F_enc + F_dec` | High-resolution details |

## Loss Functions

### Cross-Entropy Loss

```
L_CE = -Σ_k y_k log(p_k)
```

### Dice Loss

```
L_Dice = 1 - (2 × |P ∩ G| + ε) / (|P| + |G| + ε)
```

### Focal Loss

Addresses class imbalance:

```
L_Focal = -α_t (1 - p_t)^γ log(p_t)
```

| γ | Effect |
|---|--------|
| 0 | Standard CE |
| 1 | Mild down-weighting |
| 2 | Strong down-weighting (default) |
| 5 | Very strong |

### Combined Loss

```
L_total = λ_1 × L_CE + λ_2 × L_Dice + λ_3 × L_Boundary
```

Typical weights: `λ_1 = 1.0, λ_2 = 1.0, λ_3 = 0.5`

## Evaluation Metrics

### Per-Class Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| IoU | TP/(TP+FP+FN) | [0, 1] |
| Dice | 2TP/(2TP+FP+FN) | [0, 1] |
| Precision | TP/(TP+FP) | [0, 1] |
| Recall | TP/(TP+FN) | [0, 1] |

### Global Metrics

| Metric | Formula |
|--------|---------|
| mIoU | (1/K) Σ_k IoU_k |
| Pixel Accuracy | Σ_k TP_k / N |
| Mean Accuracy | (1/K) Σ_k (TP_k / N_k) |

## Benchmark Results

### EuroSAT Land Use Classification

| Model | OA (%) | Kappa |
|-------|--------|-------|
| U-Net ResNet-50 | 94.2 | 0.93 |
| DeepLabV3+ | 95.1 | 0.94 |
| Ununennium U-Net | 95.8 | 0.95 |

### DeepGlobe Land Cover

| Model | mIoU | Dice |
|-------|------|------|
| U-Net | 0.721 | 0.812 |
| FPN | 0.735 | 0.823 |
| Ununennium U-Net | 0.763 | 0.847 |

## Training Tips

1. **Class Weighting:** Use inverse frequency weights
2. **Data Augmentation:** Geometric + radiometric
3. **Multi-scale:** Training with multiple resolutions
4. **Auxiliary Loss:** Intermediate supervision

## Example

```python
from ununennium.models import create_model
from ununennium.losses import DiceLoss, CombinedLoss
import torch.nn as nn

model = create_model("unet_resnet50", in_channels=12, num_classes=10)
loss_fn = CombinedLoss([
    nn.CrossEntropyLoss(weight=class_weights),
    DiceLoss(),
], weights=[0.5, 0.5])
```
