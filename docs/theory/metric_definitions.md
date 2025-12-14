# Metric Definitions and Evaluation Theory in Geospatial AI: A Theoretical Treatise

## Abstract

Evaluation metrics are the compass by which we navigate the optimization landscape. In Geospatial AI, standard computer vision metrics (like Accuracy) are often misleading due to extreme class imbalance (the "99% Water" problem) and the geometric nature of the predictions. A model with 99% pixel accuracy can fail to detect 50% of buildings if the objects are small. This treatise rigorously defines the set-theoretic, topological, and probabilistic metrics used in Ununennium, analyzing their mathematical properties, failure modes, and optimal use-cases.

---

## 1. Confusion Matrix Primitives

Let $\mathcal{D} = \{ (y_i, \hat{y}_i) \}_{i=1}^N$ be the set of Ground Truth ($y$) and Prediction ($\hat{y}$) pairs.
For a class $c$, we define:
*   **TP (True Positive):** $y=c \land \hat{y}=c$
*   **FP (False Positive):** $y \neq c \land \hat{y}=c$ (Type I Error)
*   **FN (False Negative):** $y=c \land \hat{y} \neq c$ (Type II Error)
*   **TN (True Negative):** $y \neq c \land \hat{y} \neq c$

### 1.1 The Accuracy Paradox
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
In a dataset with 99% background and 1% target (e.g., Ships in Ocean), a trivial model predicting "All Background" achieves 99% accuracy. Thus, **Accuracy is deprecated** for all Ununennium tasks except balanced scene classification.

---

## 2. Set-Theoretic Metrics (Overlap)

These metrics view the image as a set of indices $\Omega$. Let $A = \{i | y_i = 1\}$ and $B = \{i | \hat{y}_i = 1\}$.

### 2.1 Jaccard Index (IoU - Intersection over Union)
Measures the ratio of intersection to union.

$$ \text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN} $$

*   **Range:** $[0, 1]$.
*   **Properties:** Symmetric, Scale Invariant.
*   **Hardness:** IoU penalizes errors more harshly than F1. $\text{IoU} \le \text{F1}$ always.

### 2.2 Dice Coefficient (F1 Score)
The Harmonic Mean of Precision and Recall.

$$ \text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2TP}{2TP + FP + FN} $$

*   **Use Case:** Often used as a Loss Function (Soft Dice) because it is differentiable and convex-ish.
*   **Relation to IoU:** $Dice = \frac{2 \cdot IoU}{1 + IoU}$.

---

## 3. Geometric and Boundary Metrics (Topology)

For applications like Building Footprint extraction, pixel overlap is insufficient. A model that predicts a blob covering 90% of a building gets 0.9 IoU but fails to capture the square shape.

### 3.1 Hausdorff Distance
Measures the maximum distance from a point in one set to the nearest point in the other set.

$$ d_H(A, B) = \max \left( \sup_{a \in A} \inf_{b \in B} d(a, b), \sup_{b \in B} \inf_{a \in A} d(a, b) \right) $$

*   **Interpretation:** The "worst case" error. If the model imagines a single outlier pixel 1km away, $d_H = 1km$, even if IoU $\approx 1.0$.
*   **Robustness:** Highly sensitive to outliers.

### 3.2 Boundary F1 (BF1) Score
Computes Precision and Recall only within a distance buffer $d$ of the boundaries.

$$ \text{Boundary}(S) = \{ x \in S \mid \exists y \notin S, d(x, y) < 1 \text{px} \} $$

We define a match if prediction boundary is within tolerance $\theta$ of ground truth boundary.
*   *Application:* Critical for Road Network extraction, where the width of the road changes but the centerline topology matters.

---

## 4. Detection Metrics (Object Level)

For Object Detection (Bounding Boxes), we operate on discrete objects, not pixels.

### 4.1 Average Precision (AP)
AP is the area under the Precision-Recall Curve (PR Curve).

$$ AP = \int_0^1 p(r) dr $$

In practice, we compute interpolated AP (COCO style):
$$ AP = \frac{1}{11} \sum_{r \in \{0, 0.1, ..., 1.0\}} p_{interp}(r) $$
Where $p_{interp}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$.

### 4.2 Mean Average Precision (mAP)
The mean of AP across all classes.
*   **mAP@50:** IoU threshold = 0.5.
*   **mAP@[50:95]:** Average over IoU thresholds 0.50 to 0.95 (step 0.05). Rewards tight localization.

---

## 5. Calibration Metrics (Probabilistic)

Measures the reliability of the confidence scores $\hat{p}$.

### 5.1 Brier Score
The Mean Squared Error of the probability vector.

$$ BS = \frac{1}{N} \sum_{i=1}^N ( \hat{p}_i - y_i )^2 $$

It decomposes into:
$$ BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty} $$
*   **Reliability:** How close are probabilities to true frequencies.
*   **Resolution:** How distinct are the forecasts from the global average.

### 5.2 Expected Calibration Error (ECE)
(See Uncertainty Theory for derivation).
ECE helps detect if a model is "hallucinating" high confidence on wrong answers.

---

## 6. Time Series Metrics

For comparing temporal signals $T_1$ and $T_2$.

### 6.1 Dynamic Time Warping (DTW)
Euclidean distance assumes indices align ($i$ matches $i$). DTW allows non-linear alignment (warping) to handle temporal shifts (e.g., crop season starting 10 days later).

$$ DTW(T_1, T_2) = \min_{\pi} \sqrt{ \sum_{(i,j) \in \pi} (T_1[i] - T_2[j])^2 } $$

Where $\pi$ is the warping path.

---

## 7. Aggregation Strategies (Macro vs Micro)

When calculating metrics across $N$ images and $C$ classes:

### 7.1 Micro-Averaging
Pools all pixels/objects from all images into one giant Confusion Matrix, then computes metric.
$$ F1_{micro} $$
*   **Behavior:** Dominated by frequent classes. If "Water" is 90% of pixels, Micro-F1 essentially measures Water performance.

### 7.2 Macro-Averaging
Computes metric for each class/image independently, then averages.
$$ F1_{macro} = \frac{1}{C} \sum_{c=1}^C F1_c $$
*   **Behavior:** Treats all classes equally. "Rare Bird" has same weight as "Water".
*   **Ununennium Default:** We report **Macro** metrics for validation to ensure performance on rare classes is visible.

---

## 8. Ununennium Implementation

The `ununennium.metrics` module provides a unified API backed by `torchmetrics`.

```python
class MetricCollection(nn.Module):
    def __init__(self, num_classes):
        self.iou = IoU(num_classes, reduction='elementwise_mean')
        self.f1 = F1Score(num_classes, average='macro')
        self.brier = BrierScore()
        
    def update(self, preds, target):
        self.iou.update(preds, target)
        self.f1.update(preds, target)
        
    def compute(self):
        return {
            "mIoU": self.iou.compute(),
            "MacroF1": self.f1.compute()
        }
```

---

## 9. Conclusion

A single number can never capture the performance of a complex geospatial model.
*   Use **IoU** for spatial overlap.
*   Use **Boundary F1** for shape fidelity.
*   Use **mAP** for object counting.
*   Use **ECE** for confidence reliability.

Standardizing these metrics allows us to move beyond "it looks good" to empirical science.

---

## 10. References

1.  **Powers, D. M. (2011).** "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation".
2.  **Csurka, G., et al. (2013).** "What is a good evaluation measure for semantic segmentation?". *BMVC*.
3.  **Everingham, M., et al. (2010).** "The Pascal Visual Object Classes (VOC) Challenge". *IJCV*.
4.  **Niculescu-Mizil, A., & Caruana, R. (2005).** "Predicting good probabilities with supervised learning". *ICML*.
5.  **Taha, A. A., & Hanbury, A. (2015).** "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool". *BMC Medical Imaging*.
