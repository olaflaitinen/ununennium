# Sampling Theory in Geospatial Machine Learning: A Theoretical Treatise

## Abstract

Sampling is the mechanism by which we bridge the continuous reality of the physical world and the discrete, finite datasets required for machine learning. In the geospatial domain, sampling is complicated by **Spatial Autocorrelation**, **Heterogeneity**, and **Class Imbalance**. A naive Random Sampling strategy, while theoretically sound for i.i.d. data, is often catastrophic for Earth Observation (EO) models, leading to vast redundancy, missed rare events, and inflated accuracy metrics. This document presents a comprehensive theoretical framework for Spatial Sampling Design, covering Design-based vs. Model-based inference, advanced tiling strategies, and temporal sampling requirements for dynamic phenomena.

---

## 1. The sampling Problem in Continuum

Geo-data is inherently continuous. A satellite image is a discrete discretization of a continuous radiance field $L(lat, lon, t)$. Our labels (e.g., "Forest") are discrete representations of continuous land cover gradients.

### 1.1 The "Change of Support" Problem (COSP)

The "Support" refers to the size, shape, and orientation of the physical area associated with a data value.
*   **Point Support:** A ground measurement (e.g., weather station thermometer).
*   **Areal Support:** A satellite pixel (e.g., $10m \times 10m$ average).

**The Ecological Fallacy:**
Relationships observed at one support level (e.g., correlation between NDVI and Biomass at 30m) do not necessarily hold at another support level (e.g., 1km).
*   *Implication:* A model trained on Sentinel-2 (10m) cannot simply be applied to MODIS (250m) by upsampling the input. The variance structure changes non-linearly.

---

## 2. Theoretical Frameworks of Inference

### 2.1 Design-Based Inference
*   **Assumption:** The population values $y$ are fixed constants. Randomness comes entirely from the selection probability of the sample sites $s$.
*   **Goal:** Estimate global population parameters (e.g., Total Wheat Area).
*   **Key:** Selection probabilities $\pi_i$ must be known and non-zero.
*   ** Estimator:** Horvitz-Thompson Estimator:
    $$ \hat{Y} = \sum_{i \in s} \frac{y_i}{\pi_i} $$

### 2.2 Model-Based Inference
*   **Assumption:** The observed values $y$ are realizations of a random process (spatial superpopulation).
*   **Goal:** Predict $y$ at unobserved locations (Krige/CNN predictions).
*   **Key:** The model must correctly characterize the spatial covariance structure.
*   **Role of Sampling:** To minimize the variance of the prediction error ($E[ (\hat{Y} - Y)^2 ]$).

Ununennium effectively operates in the **Model-Based** paradigm (training neural networks) but uses **Design-Based** principles for validation to ensure unbiased metric reporting.

---

## 3. Spatial Sampling Designs

How do we select $N$ locations (patches) from the study area $\mathcal{D}$?

### 3.1 Simple Random Sampling (SRS)
$$ P(x_i \in S) = \frac{1}{|\mathcal{D}|} $$
*   **Pros:** Unbiased mean estimation.
*   **Cons:**
    1.  **Clustering:** purely random points often form accidental clusters.
    2.  **Inefficiency:** Due to autocorrelation, clustered points carry redundant information.
    3.  **Gaps:** Leaves large areas unsampled.

### 3.2 Systematic Sampling (Grid)
Selects points on a regular lattice with spacing $\Delta$.
*   **Pros:** Maximizes spatial coverage (minimizes Max(min(dist))). Even distribution of error.
*   **Cons:**
    1.  **Aliasing:** If the landscape has a periodic feature (e.g., center-pivot irrigation, city blocks) matching the grid spacing, the sample will be completely biased.
    2.  **Variance Estimation:** No unbiased estimator for variance exists for a single random start systematic sample.

### 3.3 Stratified Random Sampling
Partition $\mathcal{D}$ into strata $H_1, ..., H_L$ (e.g., land cover classes: Forest, Urban, Water). Draw $n_h$ samples from each stratum.

**Neyman Optimal Allocation:**
To minimize variance of the global estimate, sample size $n_h$ should be proportional to the stratum size $N_h$ and the stratum standard deviation $\sigma_h$.

$$ n_h = n \frac{N_h \sigma_h}{\sum_{k=1}^L N_k \sigma_k} $$

*   *Implication for DL:* We should sample *more* from heterogeneous classes (Urban) and *less* from homogenous classes (Water), even if Water covers 70% of the Earth.

### 3.4 Spatially Balanced Sampling (Poisson Disk / GRTS)
**Generalized Random Tessellation Stratified (GRTS)** and **Poisson Disk** sampling generate designs that are spatially balanced (no holes, no clumps) but maintain probability properties.
*   *Algorithm (Poisson Disk):* Generate a point. Reject any subsequent point closer than radius $r$.
*   *Result:* "Blue Noise" distribution. Ideal for training set construction to force diversity.

---

## 4. Tiling Strategies for CNNs

In Deep Learning, our "samples" are not points but $H \times W$ image patches (tiles).

### 4.1 The Epoch Definition Problem
In standard computer vision (ImageNet), an "Epoch" is one pass over all images. In EO, we have one continuous image (The World). What is an epoch?
*   *Approach A (Grid):* Fixed sliding window stride $S$. Epoch = one full cover.
    *   *Drawback:* Massive redundancy ($95\%$ of ocean tiles looking identical).
*   *Approach B (Random):* $N$ random crops per "epoch".
    *   *Drawback:* No guarantee of coverage.

**Ununennium approach:**
We define an epoch as $N$ iterations, where samples are drawn via **Importance Sampling** from a probability map $P(x,y)$.

### 4.2 Importance Sampling (Hard Example Mining)

We construct a sampling probability map $M(x,y)$.

$$ M(x,y) = \alpha P_{freq}(x,y) + \beta P_{edge}(x,y) + \gamma P_{error}(x,y) $$

1.  **Class Frequency Balancing ($P_{freq}$):**
    Rare classes get higher weight.
    $$ w_c = \frac{1}{\ln(f_c + 1.2)} $$
    This "Soft Inverse Frequency" prevents over-sampling extreme outliers while boosting rare classes.

2.  **Structural Complexity ($P_{edge}$):**
    We compute the gradient magnitude $|\nabla I|$ or entropy of the image.
    *   *Logic:* Uniform green fields are easy. Edges are hard. Sample edges.

3.  **Active Learning ($P_{error}$):**
    Feed back previous epoch's loss/uncertainty.
    *   *Logic:* Focus training on areas where the model is confused (High Entropy).

### 4.3 Overlap Strategy and Edge Effects

CNNs suffer from padding artifacts at the edges of tiles.
*   **Feature Degradation:** Zero-padding distorts features near boundaries.
*   **Inference Strategy:**
    1.  **Overlap-Tile:** Extract tiles of size $D \times D$ with stride $S < D$.
    2.  **Center-Crop Prediction:** Predict on $D \times D$, but only keep the center $S \times S$.
    3.  **Gaussian Blending:** Accumulate predictions with a Gaussian weight mask falling to zero at edges.

$$ P_{final}(i, j) = \frac{\sum_k w_k(i,j) P_k(i,j)}{\sum_k w_k(i,j)} $$
Where $k$ indexes the overlapping tiles.

---

## 5. Temporal Sampling (Time Series)

For Satellite Image Time Series (SITS), the sampling dimension extends to Time $t$.

### 5.1 The Nyquist-Shannon Limit in Phenology

To classify crops, we must capture the phenological curve (NDVI trajectory).
*   **Signal:** Vegetation growth cycle (~120 days).
*   **Bandwidth:** Key inflection points (emergence, flowering, harvest) happen in ~1-2 weeks.
*   **Requirement:** Revisit rate $< \frac{1}{2} \lambda_{min} \approx 5-10$ days.

**Sentinel-2 (5 days)** satisfies this. **Landsat (16 days)** often fails, especially with cloud cover.

### 5.2 Handling Irregular Sampling

Satellite time series are never regularly sampled due to clouds.
$$ T = \{t_1, t_2, ..., t_N\}, \quad \Delta t_i \neq \text{const} $$

**Methods in Ununennium:**
1.  **Linear Interpolation (Gap Filling):** Simple, assumes linearity (bad for rapid changes).
2.  **Fourier/Harmonic Regression:** Fits sines/cosines (good for seasonality).
3.  **Attention Models (Transformers):**
    Temporal Attention mechanisms (like in PSE+TAE) naturally handle irregular positions by encoding the time differences $\Delta t$ as Positional Embeddings.

    $$ PE(t) = [\sin(\omega_1 t), \cos(\omega_1 t), ...] $$

---

## 6. The Train/Val/Test Split Geometry

Creating a valid Test set in spatial data is arguably the hardest sampling problem.

### 6.1 Spatial Leakage
If using Random Sampling (Pixel-based split):
*   Train Pixel $(i, j)$
*   Test Pixel $(i, j+1)$
*   Correlation $\approx 1.0$.
*   **Test Accuracy = 99%**, **Real World Accuracy = 60%**.

### 6.2 Block Sampling with Buffers

We must split by **Blocks** (or Scenes), separated by a **Buffer Distance**.

$$ \text{Buffer} > \text{Range of Variogram}(\alpha) $$

**Algorithm:**
1.  Compute empirical variogram of the target variable. Determine effective range $R$.
2.  Divide ROI into grid cells of size $S > R$.
3.  Randomly assign cells to Train/Val/Test.
4.  Erode the training masks by $R/2$ and validation masks by $R/2$ to create a dead zone.

### 6.3 Spatial k-Fold
Standard k-Fold is biased. We use **Spatially Blocked k-Fold**.
*   This ensures that in Fold $k$, the model is tested on a geographic region effectively unseen during training.
*   It measures **Spatial Generalization** (interpolation/extrapolation capability) rather than **memorization**.

---

## 7. Mathematical Bounds of Generalization

Learning Theory (VC-Dimension, Rademacher Complexity) assumes i.i.d.
For Spatial Mixing processes ($\beta$-mixing), the error bound loosens.

$$ E_{gen} \le E_{train} + \mathcal{O}\left( \sqrt{\frac{d \log N_{eff}}{N_{eff}}} \right) $$

where $N_{eff} \ll N$ is the effective sample size due to autocorrelation.
*   *Conclusion:* You need vastly more labeled pointers in a clustered spatial dataset to achieve the same generalization guarantee as in an i.i.d. dataset (like MNIST).

---

## 8. Implementation

Ununennium's `tiling` module provides the `GeoSampler` class hierarchy.

```python
class ConstrainedRandomGeoSampler(GeoSampler):
    """
    Samples patches such that:
    1. Intersection with ROI > threshold
    2. Cloud cover < threshold
    3. Foreground class presence > threshold
    """
    def __init__(self, dataset, prob_map, roi, size, stride):
        # Implementation uses R-Tree index for O(log N) queries
        # Rejection sampling with early exit
        pass
```

Also provided is `SpatialKFold` in `ununennium.model_selection`.

---

## 9. References

1.  **Cochran, W. G. (1977).** *Sampling Techniques*. John Wiley & Sons.
2.  **Stehman, S. V. (1999).** "Basic probability sampling designs for thematic map accuracy assessment". *International Journal of Remote Sensing*.
3.  **Olofsson, P., et al. (2014).** "Good practices for estimating area and assessing accuracy of land change". *Remote Sensing of Environment*.
4.  **Brus, D. J. (2019).** "sampling for digital soil mapping: A tutorial relevant to digital soil mapping and other domains".
5.  **Wang, J., et al. (2016).** "The spatial autocorrelation problem in the reliability of machine learning models".
