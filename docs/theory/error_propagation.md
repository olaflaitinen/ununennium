# Error Propagation and Sensitivity Analysis in Geospatial Pyplines: A Theoretical Treatise

## Abstract

In multi-stage Earth Observation (EO) pipelines, errors are not merely additive additive nuisances; they are dynamic quantities that propagate, amplify, or attenuate through non-linear transformations. A $10m$ registration error in L1C data does not simply result in a $10m$ shift in the final output; it interacts with resampling kernels, classification boundaries, and vectorization algorithms to produce complex uncertainty manifolds. This document establishes the **Calculus of Error** for Ununennium, deriving the theoretical interactions between Geodesy, Radiometry, and Deep Learning inaccuracies, and presenting frameworks for rigorous **Sensitivity Analysis**.

---

## 1. The Cascade Effect (Error Budgeting)

A typical EO pipeline $\mathcal{P}$ is a composition of functions:

$$ Y = \mathcal{P}(X) = (f_{vec} \circ f_{infer} \circ f_{norm} \circ f_{resample} \circ f_{reg})(X) $$

Where $X$ is the raw sensor data. If $X$ is perturbed by noise $\epsilon$, the output error is approximated by the Taylor expansion:

$$ \Delta Y \approx \sum_{i} \frac{\partial \mathcal{P}}{\partial x_i} \Delta x_i + \frac{1}{2} \sum_{i,j} \frac{\partial^2 \mathcal{P}}{\partial x_i \partial x_j} \Delta x_i \Delta x_j $$

### 1.1 Positional Error (Geolocation)

Satellite imagery has inherent geolocation uncertainty (RMSE).
*   **Sentinel-2:** $\approx 12m$ ($2\sigma$) without GCPs (Ground Control Points).
*   **Landsat-8:** $\approx 18m$ ($2\sigma$).

**Propagation into Classification:**
Consider a binary boundary (e.g., Coastline). Let the true boundary be at position $x_b$. The sensor reports $x_{obs} = x_b + \mathcal{N}(0, \sigma^2_{pos})$.
If we train a CNN to predict "Water" vs "Land":
1.  **Training Noise:** The label is effectively "smeared" by $\sigma_{pos}$. The CNN learns a fuzzy decision boundary of width $\approx 2\sigma_{pos}$.
2.  **Inference Error:** Even a perfect classifier will misclassify a strip of pixels along the coast with width equal to the registration error.

**Ununennium Strategy:**
We explicitly model **Aleatoric Uncertainty** at boundaries.
$$ \sigma_{pred}^2 = \sigma_{model}^2 + (\nabla I \cdot \sigma_{pos})^2 $$
Where $\nabla I$ is the image gradient. High gradient areas (edges) inherit high positional error.

---

## 2. Radiometric Error Propagation

Surface Reflectance ($\rho_{surf}$) is derived from TOA Randomness ($L_{TOA}$).

$$ \rho_{surf} = \frac{\pi (L_{TOA} - L_{path})}{E_{sun} \cos \theta T} $$

Errors in atmospheric parameters ($\Delta \tau$ - Aerosol Optical Depth) propagate non-linearly.

### 2.1 The NDVI Instability

Consider the Normalized Difference Vegetation Index:
$$ \text{NDVI} = \frac{N - R}{N + R} $$

The sensitivity to noise in the Red band ($\Delta R$) is:

$$ \frac{\partial \text{NDVI}}{\partial R} = \frac{-(N+R) - (N-R)}{(N+R)^2} = \frac{-2N}{(N+R)^2} $$

**Critical Observation:**
Sensitivity approaches infinity as $(N+R) \to 0$ (Dark targets like water/shadows).
*   *Implication:* A tiny sensor noise in a shadow pixel results in massive NDVI variance. A threshold-based water mask (`NDVI < 0`) is extremely unstable in dark regions.

---

## 3. Resampling and Aliasing Error

When reprojecting data (e.g., from UTM Zone 32 to Zone 33), we apply a resampling kernel $h(x)$.

### 3.1 Spectral Mixing Error
Bilinear interpolation creates "synthetic" spectral signatures.
Let Pixel A be "Pure Forest" ($\rho=0.2$) and Pixel B be "Pure Water" ($\rho=0.8$).
A resampled pixel located at $0.5A + 0.5B$ has $\rho=0.5$.
*   **The Artifact:** There is no material on Earth with $\rho=0.5$. It might look like "Urban" or "Soil".
*   **Result:** The classifier predicts "Building" in the middle of a lake.

### 3.2 Volume Estimation Bias
When summing pixels to estimate area (e.g., Water Volume), resampling introduces bias if the kernel is not **Flux-Conserving**.
*   *Nearest:* Unbiased expectation, high variance.
*   *Bilinear:* Biased at edges (smoothing reduces peaks).
*   *Average:* Unbiased flux, blurs shapes.

For **Carbon Stock Estimation**, Ununennium mandates **Area-Weighted Resampling** to ensure $\sum \text{Carbon}_{in} = \sum \text{Carbon}_{out}$.

---

## 4. Probabilistic Error Modeling (Monte Carlo)

Since analytical derivation of $\frac{\partial \mathcal{P}}{\partial X}$ for a Deep ResNet101 is impossible, we use **Monte Carlo Simulation**.

### 4.1 The Method of Realizations

To estimate the confidence interval of a result (e.g., "Total Deforested Area = 500 ha"):

1.  **Define Error Distributions:**
    *   $\text{Pos} \sim \mathcal{N}(0, 10m)$
    *   $\text{Rad} \sim \mathcal{N}(0, 0.01)$
    *   $\text{Atm} \sim U(0.1, 0.4)$
2.  **Generate $N$ Perturbed Inputs:**
    $X_k = X + \delta_k$
3.  **Run Pipeline $N$ times:**
    $Y_k = \mathcal{P}(X_k)$
4.  **Compute Statistics:**
    $$ \mu_Y = \frac{1}{N} \sum Y_k $$
    $$ \sigma_Y = \sqrt{\frac{1}{N-1} \sum (Y_k - \mu_Y)^2} $$

**Ununennium Implementation:**
The `ununennium.benchmarks.uncertainty` module enables wrapping any `GeoBatch` in a `StochasticTensor` which automatically forks the pipeline $N$ times.

---

## 5. Temporal Error Propagation (Time Series)

In change detection, we compare $t_1$ and $t_2$.

$$ \Delta = y_{t2} - y_{t1} $$

The variance of the difference is the sum of the variances (assuming independence):

$$ \text{Var}(\Delta) = \text{Var}(y_{t2}) + \text{Var}(y_{t1}) $$

### 5.1 Misregistration Induced Change
If $t_1$ and $t_2$ are misaligned by vector $\vec{d}$, the "False Change" signal is proportional to the local texture gradient.

$$ E_{reg} \approx \vec{d} \cdot \nabla I $$

**Pseudo-invariant Feature (PIF) Validation:**
To measure this error, we track objects that *should not change* (e.g., Roads, Large Buildings). Any $\Delta$ observed on PIFs is pure error.
Ununennium's `metrics.calibration` uses PIFs to normalize Time Series before change detection.

---

## 6. Vectorization and Topology Error

The final step is often converting probability rasters to Polygons (GeoJSON).

### 6.1 The Staircase Effect
A raster is a grid. A grid representation of a straight line at $45^\circ$ has length $L \sqrt{2}$. The true length is $L$.
*   **Perimeter Bias:** Raster perimeters always overestimate fractal coastlines.
*   **Correction:** We apply **Douglas-Peucker** simplification or use **Active Contour** losses (Level Sets) during training to encourage smooth, vector-friendly boundaries.

### 6.2 Topological Violations
A pixel-wise Argmax can produce:
1.  **Holes:** Single pixel dropouts inside a building.
2.  **Islands:** Single pixel noise in the ocean.
3.  **Self-Intersection:** (During polygonization).

**Morphological Post-Processing:**
We apply `Opening` (Erosion $\circ$ Dilation) and `Closing` (Dilation $\circ$ Erosion) to enforce topological consistency.
$$ Y_{clean} = \text{Close}(\text{Open}(Y_{raw}, K), K) $$

---

## 7. Validated Error Bounds (Olofsson's Protocol)

For Area Estimation (e.g., REDD+ Reporting), simply counting pixels is scientifically invalid. We must use the **Adjusted Estimator**.

### 7.1 Stratified Estimator
Using the Confusion Matrix $P_{ij}$ (proportion of area of class $i$ mapped as class $j$):

$$ \hat{A}_k = A_{total} \sum_{i} W_i \frac{n_{ik}}{n_{i.}} $$

*   $W_i$: Mapped proportion of class $i$.
*   $n_{ik}$: Validation samples mapped as $i$ but truly $k$.

This estimator corrects for the bias of the classifier (e.g., if the model systematically over-predicts Forest).

---

## 8. Conclusion

Error in EO is inevitable. High-fidelity modeling does not mean eliminating error, but **bounding** it.
*   **Positional Error** limits the smallest detectable feature.
*   **Radiometric Error** limits the subtlest detectable change.
*   **Sampling Error** limits the confidence of area estimates.

Ununennium provides the primitives to not just predict $Y$, but to report $Y \pm \delta$, elevating Deep Learning from a black-box oracle to a calibrated scientific instrument.

---

## 9. References

1.  **Olofsson, P., et al. (2014).** "Good practices for estimating area and assessing accuracy of land change". *Remote Sensing of Environment*.
2.  **Congalton, R. G. (1991).** "A review of assessing the accuracy of classifications of remotely sensed data". *Remote Sensing of Environment*.
3.  **Foody, G. M. (2002).** "Status of land cover classification accuracy assessment". *Remote Sensing of Environment*.
4.  **McRoberts, R. E. (2011).** "Satellite image-based maps: Scientific inference or pretty pictures?". *Remote Sensing of Environment*.
5.  **Heuvelink, G. B. (1998).** *Error propagation in environmental modelling with GIS*. Taylor & Francis.
