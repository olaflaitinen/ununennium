# Signal Processing and Resampling Kernels in Remote Sensing: A Theoretical Treatise

## Abstract

Resampling is the digital signal processing operation of transforming a discrete sampled signal from one coordinate system to another. In Deep Learning for Remote Sensing, this occurs constantly: reprojecting maps to common CRS, resizing images for fixed-size CNN inputs, or aligning multi-temporal acquisitions. While often treated as a trivial implementation detail (`cv2.resize`), the choice of the resampling kernel $h(x)$ has profound spectral and spatial implications. Poor resampling introduces Aliasing (Moiré patterns), Ringing (Gibbs phenomenon), and Spectral Mixing, which establish hard ceilings on model performance. This treatise explores the Fourier Theory of resampling and guides the selection of optimal kernels for EO data.

---

## 1. The Sampling Theorem and Reconstruction

A continuous signal $f(x)$ is sampled at intervals $T$ to create a discrete sequence $f[n] = f(nT)$. To recover or resample this signal at a new point $x'$, we must perform **Reconstruction**.

$$ \hat{f}(x) = \sum_{n=-\infty}^{\infty} f[n] \cdot h\left(\frac{x - nT}{T}\right) $$

Here, $h(t)$ is the **Reconstruction Kernel**.Ideally, to perfectly reconstruct a band-limited signal, $h(t)$ must be the normalized sinc function (Whittaker–Shannon interpolation formulation):

$$ h(t) = \operatorname{sinc}(t) = \frac{\sin(\pi t)}{\pi t} $$

However, the sinc function has infinite support (extends to $\pm \infty$), making it computationally impossible. All practical kernels are finite approximations of the sinc.

---

## 2. Kernel Taxonomy in Frequency Domain

We analyze kernels based on their spatial shape $h(x)$ and their Frequency Response $H(f) = \mathcal{F}\{h(x)\}$.

### 2.1 Nearest Neighbor (Box Kernel)

The 0th-order interpolator.

$$ h(t) = \begin{cases} 1 & |t| \le 0.5 \\ 0 & \text{otherwise} \end{cases} $$

*   **Frequency Response:** $H(f) = \operatorname{sinc}(f)$.
*   **analysis:**
    *   **Passband:** Poor. It begins attenuating frequencies well before the Nyquist limit.
    *   **Stopband:** Terrible. It has infinite side-lobes that allow high-frequency aliases to fold back into the signal.
*   **Artifacts:** "Jaggies" (aliasing), Blockiness, sub-pixel spatial shifts ($\pm 0.5$ px jitter).
*   **Use Case:** **Categorical Masks**. We cannot interpolate the integer "Class 5 (Forest)" and "Class 1 (Water)" to get "Class 3 (Urban)". We must strictly preserve values.

### 2.2 Bilinear Interpolation (Triangle Kernel)

The 1st-order interpolator. Convolution of two box kernels.

$$ h(t) = \begin{cases} 1 - |t| & |t| < 1 \\ 0 & \text{otherwise} \end{cases} $$

*   **Frequency Response:** $H(f) = \operatorname{sinc}^2(f)$.
*   **Analysis:**
    *   The squared sinc decays faster ($1/f^2$), reducing aliasing compared to Nearest.
    *   **Low-Pass Filter:** It strongly attenuates high frequencies within the passband. This results in **Blurring**.
*   **Use Case:** fast on-the-fly resizing where sharpness is secondary to speed.

### 2.3 Bicubic Convolution (Keys' Cubic)

A 3rd-order polynomial approximation of the sinc. Defined by a parameter $\alpha$ (usually -0.5 or -0.75, which matches the slope of the sinc function at $x=1$).

$$ h(t) = \begin{cases} (\alpha+2)|t|^3 - (\alpha+3)|t|^2 + 1 & |t| < 1 \\ \alpha|t|^3 - 5\alpha|t|^2 + 8\alpha|t| - 4\alpha & 1 \le |t| < 2 \\ 0 & \text{otherwise} \end{cases} $$

*   **Analysis:**
    *   Sharper passband than bilinear.
    *   Better stopband suppression.
    *   **Negative Lobes:** The kernel goes negative. This can produce output values outside the input range (Over/Undershoot).
*   **Artifacts:** **Ringing (Gibbs Phenomenon)**. High-contrast edges (e.g., coastline) will have a "halo" or "ghost" line parallel to them.
*   **Use Case:** Visualization (RGB), DEMs (requires continuous derivatives).

### 2.4 Lanczos Resampling

A Windowed Sinc function. $\operatorname{sinc}(x)$ multiplied by a "Lanczos Window" (central lobe of a larger sinc).

$$ L(x) = \begin{cases} \operatorname{sinc}(x) \operatorname{sinc}(x/a) & -a < x < a \\ 0 & \text{otherwise} \end{cases} $$

Usually $a=3$ (Lanczos-3).
*   **Analysis:** The closest practical approximation to the ideal low-pass filter.
*   **Pros:** Extreme sharpness, minimal aliasing.
*   **Cons:** Computatioanlty expensive (larger support window), still suffers from Ringing.

---

## 3. Aliasing and The Nyquist Limit

**Aliasing** occurs when we downsample (decimate) a signal without removing frequencies higher than the new Nyquist rate ($f_{new} = f_{old} / \text{scale}$).

*   **Spatial Moiré:** High-frequency patterns (e.g., rows of crops, tile roofs) create low-frequency interference patterns when resized carelessly.
*   **Deep Learning Impact:** CNNs are texture-biased. If a Corn field (periodic texture) is downsampled aliased to look like a swamp (noisy texture), the model fails.

**The Golden Rule of Downsampling:**
Always apply a Low-Pass Filter (Blur) *before* decimation to remove frequencies $> f_{new}$.
*   **Average / Area-Weighted Resampling:** This is mathematically equivalent to integration (box filter) followed by sampling. It conserves the total radiometric energy (Flux).
*   **Ununennium Standard:** For downscaling satellite imagery (e.g., converting 10m Sentinel to 30m Landsat grid), we strictly use **Average** or **Gauss** resampling mechanisms to preserve physical flux.

---

## 4. Phase Shift Errors (The "Shift" Problem)

In Nearest Neighbor resampling, the grid snaps to the closest integer.
$$ x_{new} = \operatorname{round}(x_{old}) $$
This introduces a random shift error $\epsilon \sim U[-0.5, 0.5]$ pixels.

In **Change Detection** (Siamese Networks), this is fatal.
*   Image T1 (Jan): Shifted -0.4 px.
*   Image T2 (Feb): Shifted +0.4 px.
*   Total Misalignment: 0.8 px.
*   Result: False positive "Change" detected along every distinct edge in the image.

**Solution:** Ununennium's `GeoTensor` operations use sub-pixel precise alignment utilizing Bilinear/Bicubic kernels even for small shifts, strictly avoiding Nearest Neighbor for co-registration tasks.

---

## 5. Spline Interpolation

Beyond convolution kernels, we can fit piecewise polynomials (Splines) to the data points.
**B-Splines** provide maximum smoothness ($C^n$ continuity).
*   *Application:* Creating Digital Elevation Models (DEMs) from sparse LiDAR points.
*   *Relevance:* Calculating Slope and Aspect (derivatives of elevation) requires the surface to be differentiable. Nearest Neighbor produces valid elevation but garbage slope (0 or infinity). Cubic Splines ensure smooth derivatives.

---

## 6. Implementation in Ununennium

We wrap both `rasterio` (GDAL) and `torch.nn.functional` resampling modes.

### 6.1 `GeoTensor.reproject()`

Automatically selects the kernel based on the semantic type of the tensor (stored in metadata).

| Tensor Type | Kernel | Rationale |
|-------------|--------|-----------|
| `Mask` (Class ID) | `nearest` | Preserves integers. |
| `Mask` (Probability) | `bilinear` | Smooth probability field. |
| `Image` (Optical) | `cubic` | Visual sharpness. |
| `Image` (Radiance) | `average` | Energy conservation (if downsampling). |
| `SAR` (Db) | `average` | Noise reduction (multilooking). |

### 6.2 The `AreaWeighted` Resampler

For flux-conserving transformations (e.g., Population Count rasters, Emission grids):

```python
def reproject_flux(tensor, src_transform, dst_transform):
    """
    Ensures sum(input) == sum(output).
    Standard interpolation preserves density (value), not sum.
    """
    # ... Implementation utilizing fractional overlap calculation ...
```

---

## 7. Conclusion

Resampling is not just "making pixels bigger or smaller". It is a filtering operation that fundamentally alters the spectral and spatial content of the data. Use **Nearest** only for masks. Use **Average** for downscaling physics. Use **Lanczos/Cubic** for upscaling/visualization. Ignoring this leads to aliased training data, shift artifacts, and physically impossible spectral values.

---

## 8. References

1.  **Shannon, C. E. (1949).** "Communication in the Presence of Noise". *Proceedings of the IRE*.
2.  **Keys, R. (1981).** "Cubic convolution interpolation for digital image processing". *IEEE TASSP*.
3.  **Turkowski, K. (1990).** "Filters for common resampling tasks". *Graphics Gems*.
4.  **Parker, J. A., et al. (1983).** "Comparison of interpolating methods for image resampling". *IEEE TMN*.
