# Sampling Theory

Mathematical foundations for spatial sampling in remote sensing.

## Nyquist-Shannon Sampling Theorem

For a band-limited signal with maximum frequency `f_max`, alias-free reconstruction requires:

```
f_s ≥ 2 × f_max
```

**Spatial interpretation:**
- `GSD` (Ground Sample Distance) ≤ `λ_min / 2`
- Where `λ_min` is the smallest spatial feature of interest

## Spatial Resolution and GSD

### Ground Sample Distance

```
GSD = (H × p) / f
```

Where:
- `H` = altitude (m)
- `p` = pixel pitch (m)
- `f` = focal length (m)

### Resolution Types

| Type | Definition | Example |
|------|------------|---------|
| Spatial | Ground area per pixel | 10m × 10m |
| Spectral | Number of bands | 13 bands |
| Radiometric | Bits per pixel | 12-bit |
| Temporal | Revisit frequency | 5 days |

## Aliasing Effects

When sampling below Nyquist:

```
f_alias = |f_signal - n × f_s|
```

**Visual effects:**

| Pattern | Cause | Mitigation |
|---------|-------|------------|
| Moiré | High-frequency structure | Anti-aliasing filter |
| Jagged edges | Undersampling | Super-resolution |
| Mixed pixels | Sub-pixel features | Spectral unmixing |

## Point Spread Function (PSF)

The system PSF is the convolution of all components:

```
PSF_total = PSF_optics ⊛ PSF_detector ⊛ PSF_motion
```

**Modulation Transfer Function (MTF):**
```
MTF(f) = |ℱ{PSF}(f)|
```

### MTF at Nyquist

| Sensor | MTF @ Nyquist | Quality |
|--------|---------------|---------|
| Sentinel-2 B02 | 0.15 | Good |
| Landsat-8 Pan | 0.10 | Acceptable |
| WorldView-3 | 0.25 | Excellent |

## Effective Resolution

Accounting for atmospheric effects:

```
R_eff = √(R_sensor² + R_atm² + R_motion²)
```

## Optimal Sampling Strategies

### Regular Grid

```
x_i = x_0 + i × Δx
y_j = y_0 + j × Δy
```

**Pros:** Simple, efficient
**Cons:** May miss periodic structures

### Stratified Random

Each stratum receives `n/k` samples:

```
Sample_stratum_i ~ Uniform(bounds_i)
```

**Variance reduction:**
```
Var_stratified ≤ Var_simple
```

### Latin Hypercube

Ensures each row and column has exactly one sample:

```
x_i = (π(i) + U_i) / n
```

Where `π` is a random permutation and `U ~ Uniform(0, 1)`.

## Recommendations

| Scenario | Sampling Strategy | Sample Rate |
|----------|-------------------|-------------|
| Urban areas | Regular grid | 2× Nyquist |
| Agriculture | Stratified | 1.5× Nyquist |
| Forest | Random | 1.5× Nyquist |
| Mixed | Latin Hypercube | 2× Nyquist |
