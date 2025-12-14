# Geodesic Calculations in Earth Observation: A Theoretical Treatise

## Abstract

The fundamental canvas of Earth Observation is not a flat plane, but an irregular, dynamic oblate spheroid. The naive assumption of Euclidean geometry—standard in most Computer Vision libraries—introduces catastrophic metric errors when applied to planetary scales. This document details the rigorous Geodesic Mathematics underpinning the Ununennium library. We explore the evolution of Earth models from spherical approximations to modern geoids, derive the complete solution to the Direct and Inverse Geodesic Problems (Vincenty’s and Karney’s algorithms), and analyze the non-trivial implications of ellipsoidal geometry on Deep Learning architectures, specifically regarding area distortion, directional convolution, and metric learning.

---

## 1. The Shape of the Earth: From Sphere to Geoid

### 1.1 The Spherical Approximation
Historically, for navigation over short distances, Earth was approximated as a sphere of radius $R \approx 6371 \text{km}$.
*   **Meridional Radius:** Constant.
*   **Metric:** Simple Spherical Trigonometry (Great Circles).
*   **Error:** At $45^\circ$ latitude, the difference between a sphere and the true ellipsoid is approx 21km in radius. This results in distance errors of up to 0.5% (~5m per km), which is unacceptable for cadastral surveying or precision agriculture.

### 1.2 The Oblate Spheroid (Ellipsoid of Revolution)
Due to rotation, centrifugal force causes the Earth to bulge at the equator and flatten at the poles. Ideally, this shape is an **Oblate Spheroid**.

Defined by two parameters:
1.  **Semi-major axis ($a$):** Equatorial radius.
2.  **Semi-minor axis ($b$):** Polar radius.

**Flattening ($f$):**
$$ f = \frac{a - b}{a} $$

**First Eccentricity ($e$):**
$$ e^2 = \frac{a^2 - b^2}{a^2} = 2f - f^2 $$

**Second Eccentricity ($e'$):**
$$ e'^2 = \frac{a^2 - b^2}{b^2} = \frac{e^2}{1-e^2} $$

#### Standard Ellipsoids
| Ellipsoid | $a$ (meters) | $1/f$ | Usage |
|-----------|--------------|-------|-------|
| **WGS84** | 6378137.0 | 298.257223563 | GPS, Global Standard |
| **GRS80** | 6378137.0 | 298.257222101 | North America (NAD83) |
| **Airy 1830**| 6377563.4 | 299.3249646 | UK (OSGB36) |
| **Krassovsky**| 6378245.0 | 298.3 | Russia, Eastern Bloc |

Ununennium defaults to **WGS84** unless strictly specified otherwise by the CRS.

### 1.3 The Geoid (The Gravity Model)
The ellipsoid is a geometric abstraction. The physical Earth is defined by gravity. The **Geoid** is the equipotential surface of the Earth's gravity field that equates to Mean Sea Level (MSL) if the oceans were at rest and extended through the continents.

$$ h = H + N $$
*   **$h$ (Ellipsoidal Height):** Native height from GPS (geometric).
*   **$H$ (Orthometric Height):** Height above Sea Level (physical/gravitational).
*   **$N$ (Geoid Undulation):** The separation between Ellipsoid and Geoid.

$N$ ranges from -106m (in the Indian Ocean) to +85m (near Iceland).
**Deep Learning Implication:**
When training models on DSM/DTM (Digital Surface Models), mixing ellipsoidal heights (from raw LiDAR) with orthometric heights (from contour maps) creates significant bias. Ununennium's `io` module checks for vertical datum compatibility (e.g., EGM96 vs EGM2008).

---

## 2. The Geodesic Problem

The shortest path between two points on an ellipsoid is a **Geodesic**. Unlike a Great Circle on a sphere, a geodesic on an ellipsoid does not return to its starting point; it oscillates between maximum northern and southern latitudes.

### 2.1 The Metric Tensor
The differential distance $ds$ on the surface is given by the First Fundamental Form:

$$ ds^2 = M(\phi)^2 d\phi^2 + N(\phi)^2 \cos^2\phi d\lambda^2 $$

Where the radii of curvature are:
*   **Meridional Radius ($M$):** Radius of curvature along the meridian (North-South).
    $$ M(\phi) = \frac{a(1-e^2)}{(1-e^2 \sin^2\phi)^{3/2}} $$
*   **Prime Vertical Radius ($N$):** Radius of curvature perpendicular to the meridian (East-West).
    $$ N(\phi) = \frac{a}{\sqrt{1-e^2 \sin^2\phi}} $$

Notably, $M(\phi)$ varies by ~1% from pole to equator. A "degree of latitude" is longer at the poles ($111.7$ km) than at the equator ($110.6$ km). This 1.1km discrepancy per degree is the primary source of error in spherical assumptions.

---

## 3. Distance Algorithms

### 3.1 The Haversine (Spherical)
Fast, differentiable, but inaccurate ($0.5\%$ error).

$$ \theta = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)} $$
$$ d = R \cdot \theta $$

Used in Ununennium only for "Quick Look" operations or lightweight neighbor search.

### 3.2 Vincenty's Formulae (Iterative Ellipsoidal)
Vincenty (1975) proposed an iterative solution to the inverse problem accurate to $0.5$ mm.

**Algorithm (Inverse Problem):**
Given $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$, find distance $s$ and azimuths $\alpha_1, \alpha_2$.

1.  Compute reduced latitudes: $U_1 = \arctan((1-f)\tan\phi_1)$, $U_2 = \arctan((1-f)\tan\phi_2)$.
2.  Initialize longitude difference on auxiliary sphere $\Lambda = \lambda_2 - \lambda_1$.
3.  **Iterate until convergence ($\Delta \Lambda < 10^{-12}$):**
    *   $\sin\sigma = \sqrt{(\cos U_2 \sin\Lambda)^2 + (\cos U_1 \sin U_2 - \sin U_1 \cos U_2 \cos \Lambda)^2}$
    *   $\cos\sigma = \sin U_1 \sin U_2 + \cos U_1 \cos U_2 \cos \Lambda$
    *   $\sigma = \arctan2(\sin\sigma, \cos\sigma)$
    *   $\sin\alpha = \frac{\cos U_1 \cos U_2 \sin \Lambda}{\sin\sigma}$
    *   $\cos^2\alpha = 1 - \sin^2\alpha$
    *   $\cos(2\sigma_m) = \cos\sigma - \frac{2\sin U_1 \sin U_2}{\cos^2\alpha}$ (handle division by zero if equatorial)
    *   $C = \frac{f}{16} \cos^2\alpha [4 + f(4 - 3\cos^2\alpha)]$
    *   $\Lambda_{new} = L + (1-C)f\sin\alpha \{ \sigma + C\sin\sigma [\cos(2\sigma_m) + C\cos\sigma(-1 + 2\cos^2(2\sigma_m))] \}$

4.  Calculate $s$ using Bessel's formula corrections.

**Limitations:**
Vincenty's iteration fails to converge for nearly antipodal points (opposite sides of the Earth).

### 3.3 Karney's Algorithm (2013)
Charles Karney solved the antipodal convergence problem using a generalized series expansion.
*   **Accuracy:** 15 nm (nanometers).
*   **Ununennium Implementation:** We wrap the `geographiclib` C++ implementation via `pyproj` for maximum speed and stability. All critical `measure_distance` calls in the library use Karney's method.

---

## 4. Area Calculations and Distortion

Calculating area on the ellipsoid requires integration of the Gaussian curvature.

### 4.1 Polygon Area
The area of a geodesic polygon is given by the discrete sum of excesses.
$$ A = \sum_{edges} (Area under geodesic segment) $$
This uses the Danielsen (1989) integrals involving Fourier series expansions of the oscillatory terms.

### 4.2 Map Projection Distortion
Deep Learning models (CNNs) operate on raster grids, which are planar projections of the ellipsoid.
$$ (x, y) = P(\phi, \lambda) $$

**Tissot's Indicatrix** describes the distortion at any point.
1.  **Conformal Projections (e.g., Mercator, stereographic):**
    *   Preserve Angles (Shapers). A circle on Earth $\to$ Circle on Map.
    *   Distort Area. $Scale \propto \sec(\phi)$.
    *   *Issue:* A pixel at 60°N represents $4\times$ less real-world area than a pixel at the Equator. A "House" class at 60°N looks 4x bigger (in pixels) than a "House" at the Equator.

2.  **Equal-Area Projections (e.g., Albers Conic, Gall-Peters):**
    *   Preserve Area.
    *   Distort Shape (Shearing).
    *   *Issue:* Objects are squashed or stretched. Rotation-variant CNNs may fail to recognize a squashed building.

**Ununennium Strategy:**
We strongly recommend **Equal-Area** projections (like Albers or Sinusoidal) for statistical reporting (e.g., "Total Deforested Area"). For training, if the AOI is global, we implement **Dynamic Scale Augmentation** or **Latitude-Weighted Loss**.

#### latitude-Weighted Loss
To counter Mercator distortion in global models:
$$ w(pixel) = \frac{A_{true}}{A_{map}} = \cos(\phi) $$
We downweight high-latitude pixels because 1 "map meter" corresponds to fewer "real meters".

---

## 5. Coding Operations and Rhumb Lines

### 5.1 Rhumb Lines (Loxodromes)
A path of constant bearing (azimuth). On a Mercator projection, this is a straight line.
*   **Geodesic:** Shortest path (Curved on map).
*   **Rhumb Line:** Constant compass heading (Straight on map).

**Use Case in AI:**
Tracking ships in AIS (Automatic Identification System) data. Ships often follow Rhumb lines (constant heading) rather than Geodesics (requires constant steering adjustment). Predicting vessel trajectories using LSTM/Transformers requires discerning the navigation mode.

---

## 6. Implementation in `geotensor`

The `GeoTensor` class enables algebraic operations that respect geodetic reality.

### 6.1 The Geodesic Buffer
`GeoTensor.buffer(distance_meters)`
Naive buffering (dilating by $N$ pixels) creates circles of varying physical sizes depending on latitude.
Ununennium computes the buffer in the Geodetic space and rasterizes it back, ensuring a 50m buffer is exactly 50m everywhere.

### 6.2 Bearing Calculation
`GeoTensor.bearing_to(other_tensor)`
Calculates the forward azimuth $\alpha_1$ for every pixel to a target.
*   **Input:** Two $(B, H, W)$ tensors of lat/lons.
*   **Output:** One $(B, H, W)$ tensor of bearings $[0, 360)$.
*   **Math:** Uses vectorised `atan2` on the sphere (or Vincenty for precision).

$$ \theta = \operatorname{atan2}(\sin\Delta\lambda \cdot \cos\phi_2, \cos\phi_1 \cdot \sin\phi_2 - \sin\phi_1 \cdot \cos\phi_2 \cdot \cos\Delta\lambda) $$

---

## 7. Performance Considerations

Ellipsoidal math is computationally expensive.
*   **Haversine:** ~20 FLOPS.
*   **Vincenty:** ~2000 FLOPS (Iterative).
*   **Karney:** ~3000 FLOPS.

**Batching Strategy:**
For heavy metric learning (e.g., Geodesic Triplet Loss), calculating exact geodesic distances matrix-wise $(N \times N)$ is too slow.
**Approximation:**
We project local batches to an **Azimuthal Equidistant Projection** centered on the batch centroid. In this local projected space, Euclidean distance $\approx$ Geodesic distance.
$$ d_{geo}(x, y) \approx \| P_{local}(x) - P_{local}(y) \|_2 $$
This reduces complexity to $O(1)$ per pair, allowing massive batch sizes.

---

## 8. Conclusion

Geometry is the physics of geospatial data. Ununennium refuses to treat Earth as a plane. By implementing rigorous WGS84 ellipsoidal mathematics (Karney/Vincenty) and handling projection distortions explicitly, we ensure that the Deep Learning models we build are not just pattern matchers, but physical measurement instruments.

---

## 9. References

1.  **Vincenty, T. (1975).** "Direct and Inverse Solutions of Geodesics on the Ellipsoid with application of nested equations". *Survey Review*, 23(176), 88-93.
2.  **Karney, C. F. F. (2013).** "Algorithms for geodesics". *Journal of Geodesy*, 87(1), 43-55.
3.  **Snyder, J. P. (1987).** *Map Projections: A Working Manual*. USGS Professional Paper 1395.
4.  **Danielsen, J. (1989).** "The Area under the Geodesic". *Survey Review*, 30(232), 61-66.
5.  **Tissot, N. A. (1859).** *Mémoire sur la représentation des surfaces et les projections des cartes géographiques*.
6.  **Rapp, R. H. (1991).** *Geometric Geodesy, Part I*. Ohio State University Department of Geodetic Science.
