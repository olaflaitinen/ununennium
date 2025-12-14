# Metric Definitions

Comprehensive reference for evaluation metrics in remote sensing machine learning.

## Classification Metrics

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation:**
- Range: [0, 1]
- Higher is better
- Limitation: Misleading for imbalanced classes

### Precision and Recall

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

| Metric | Focus | Use When |
|--------|-------|----------|
| Precision | False positives matter | Cost of false alarm is high |
| Recall | False negatives matter | Missing detection is costly |

### F1-Score

Harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Cohen's Kappa (κ)

Agreement beyond chance:

```
κ = (p_o - p_e) / (1 - p_e)
```

Where:
- `p_o` = observed agreement
- `p_e` = expected agreement by chance

**Interpretation:**

| κ Value | Agreement |
|---------|-----------|
| < 0 | Less than chance |
| 0.0 - 0.20 | Slight |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost perfect |

## Segmentation Metrics

### Intersection over Union (IoU / Jaccard Index)

```
IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN)
```

**Per-class IoU:**
```python
def iou_per_class(pred, target, num_classes):
    iou = []
    for c in range(num_classes):
        intersection = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        iou.append(intersection / union)
    return iou
```

### Mean IoU (mIoU)

```
mIoU = (1/K) × Σ_k IoU_k
```

Where K = number of classes.

### Dice Coefficient (F1 for Segmentation)

```
Dice = 2|A ∩ B| / (|A| + |B|) = 2TP / (2TP + FP + FN)
```

**Relationship to IoU:**
```
Dice = 2 × IoU / (1 + IoU)
IoU = Dice / (2 - Dice)
```

### Pixel Accuracy

```
PA = Σ_k n_kk / Σ_k Σ_l n_kl
```

**Class-weighted Pixel Accuracy:**
```
wPA = (1/K) × Σ_k (n_kk / Σ_l n_kl)
```

### Boundary IoU

IoU computed only on boundary pixels:

```
Boundary IoU = |B_pred ∩ B_true| / |B_pred ∪ B_true|
```

Where B = morphological boundary.

## Reconstruction Metrics

### Peak Signal-to-Noise Ratio (PSNR)

```
PSNR = 10 × log₁₀(MAX² / MSE)
```

Where MAX = maximum pixel value (255 for 8-bit).

**Typical values:**

| PSNR (dB) | Quality |
|-----------|---------|
| < 20 | Poor |
| 20 - 30 | Acceptable |
| 30 - 40 | Good |
| > 40 | Excellent |

### Structural Similarity Index (SSIM)

```
SSIM(x, y) = [l(x,y)]^α × [c(x,y)]^β × [s(x,y)]^γ
```

Where:
- `l` = luminance comparison
- `c` = contrast comparison
- `s` = structure comparison

```
l(x,y) = (2μ_x μ_y + C₁) / (μ_x² + μ_y² + C₁)
c(x,y) = (2σ_x σ_y + C₂) / (σ_x² + σ_y² + C₂)
s(x,y) = (σ_xy + C₃) / (σ_x σ_y + C₃)
```

### Spectral Angle Mapper (SAM)

For multi-spectral imagery:

```
SAM = arccos(Σᵢ xᵢyᵢ / (√(Σᵢ xᵢ²) × √(Σᵢ yᵢ²)))
```

**Interpretation:**
- Range: [0, π/2]
- Lower is better
- Units: radians

## Calibration Metrics

### Expected Calibration Error (ECE)

```
ECE = Σ_m (|B_m| / n) × |acc(B_m) - conf(B_m)|
```

**Interpretation:**

| ECE | Calibration |
|-----|-------------|
| < 0.05 | Excellent |
| 0.05 - 0.10 | Good |
| 0.10 - 0.20 | Fair |
| > 0.20 | Poor |

### Reliability Diagram

Plot of accuracy vs. confidence across bins.

| Property | Perfect | Under-confident | Over-confident |
|----------|---------|-----------------|----------------|
| Curve position | On diagonal | Above diagonal | Below diagonal |

## Uncertainty Metrics

### Negative Log-Likelihood (NLL)

```
NLL = -log p(y | x, θ)
```

### Brier Score

```
BS = (1/N) × Σᵢ (pᵢ - yᵢ)²
```

Where:
- `p` = predicted probability
- `y` = true label (0 or 1)

## Summary Table

| Metric | Range | Direction | Balanced | Probabilistic |
|--------|-------|-----------|----------|---------------|
| Accuracy | [0, 1] | ↑ | No | No |
| mIoU | [0, 1] | ↑ | Yes | No |
| Dice | [0, 1] | ↑ | Yes | No |
| PSNR | [0, ∞) | ↑ | N/A | No |
| SSIM | [-1, 1] | ↑ | N/A | No |
| SAM | [0, π/2] | ↓ | N/A | No |
| ECE | [0, 1] | ↓ | N/A | Yes |
| Kappa | [-1, 1] | ↑ | Yes | No |
