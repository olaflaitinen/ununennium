# Uncertainty Quantification

Methods for estimating and quantifying prediction uncertainty.

## Types of Uncertainty

### Aleatoric Uncertainty

**Definition:** Inherent randomness in the data, irreducible.

**Sources:**
- Sensor noise
- Atmospheric variability
- Sub-pixel mixing

**Modeling:**
```
p(y|x, θ) = N(μ(x), σ²(x))
```

Where the network predicts both `μ` and `σ²`.

### Epistemic Uncertainty

**Definition:** Model uncertainty due to limited data, reducible.

**Sources:**
- Limited training data
- Distribution shift
- Model misspecification

**Behavior:**
```
lim_{n→∞} Epistemic Uncertainty → 0
```

## Methods

### Monte Carlo Dropout

Apply dropout at test time:

```
y_mean = (1/T) × Σ_t f(x; θ, z_t)
y_var = (1/T) × Σ_t (f(x; θ, z_t) - y_mean)²
```

Where `z_t` are dropout masks and `T` is number of samples.

| Samples (T) | Quality | Overhead |
|-------------|---------|----------|
| 5 | Poor | 5× |
| 10 | Acceptable | 10× |
| 30 | Good | 30× |
| 100 | Excellent | 100× |

### Deep Ensembles

Train multiple models independently:

```
y_mean = (1/M) × Σ_m f_m(x)
y_var = (1/M) × Σ_m (f_m(x) - y_mean)²
```

**Advantages:**
- Simple to implement
- Parallelizable
- State-of-the-art calibration

**Disadvantages:**
- M× training cost
- M× inference cost
- Storage overhead

### Heteroscedastic Loss

Train to predict mean and variance:

```
L = (1/2σ²)(y - μ)² + (1/2)log(σ²)
```

### Bayesian Neural Networks

Place priors on weights:

```
p(θ|D) ∝ p(D|θ) × p(θ)
```

**Inference methods:**

| Method | Accuracy | Speed |
|--------|----------|-------|
| MCMC | Exact | Very slow |
| Variational | Approximate | Moderate |
| Laplace | Approximate | Fast |

## Calibration

### Temperature Scaling

Post-hoc calibration:

```
p_calibrated = softmax(z / T)
```

Where `T > 1` reduces overconfidence.

### Platt Scaling

Logistic regression on validation set:

```
p = σ(a × z + b)
```

## Practical Recommendations

| Scenario | Method | Reason |
|----------|--------|--------|
| Production | MC Dropout (T=10) | Balance speed/quality |
| Research | Deep Ensembles (M=5) | Best calibration |
| Limited compute | Temperature Scaling | Minimal overhead |
| Safety-critical | Ensembles + Calibration | Maximum reliability |

## Visualization

### Uncertainty Maps

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(prediction)
plt.title("Prediction")
plt.subplot(132)
plt.imshow(aleatoric_unc, cmap='hot')
plt.title("Aleatoric Uncertainty")
plt.subplot(133)
plt.imshow(epistemic_unc, cmap='hot')
plt.title("Epistemic Uncertainty")
```
