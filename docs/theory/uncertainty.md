# Uncertainty Quantification in Geospatial AI: A Theoretical Treatise

## Abstract

In safety-critical EO applications—such as disaster response, autonomous aerial navigation, and climate policy enforcement—a point prediction ($\hat{y}$) is insufficient and potentially dangerous. Decision-makers require a rigorous quantification of confidence. However, standard Deep Neural Networks (DNNs) are notoriously overconfident, often assigning $99\%$ probability to incorrect predictions on out-of-distribution (OOD) data. This document explores the theoretical decomposition of uncertainty into Aleatoric and Epistemic components and details the advanced Bayesian and conformal prediction methods implemented in Ununennium to calibrate these estimates.

---

## 1. The Anatomy of Uncertainty

Formally, given input $x$ and output $y$, we seek the predictive posterior distribution $P(y|x, D)$, where $D$ is the training data.

$$ P(y|x, D) = \int P(y|x, \theta) P(\theta|D) d\theta $$

This marginalization over parameters $\theta$ is intractable for modern DNNs. We approximate detailed components:

### 1.1 Aleatoric Uncertainty (Data Noise)
The uncertainty inherent in the observation process. Even with infinite data and a perfect model, this cannot be reduced.
*   **Homoscedastic:** Constant noise variance $\sigma^2$ (e.g., thermal sensor noise).
*   **Heteroscedastic:** Noise variance $\sigma(x)^2$ depends on input.
    *   *Example:* Clouds, shadows, or mixed pixels (land/water border) have higher ambiguity than clear pixels.

### 1.2 Epistemic Uncertainty (Model Knowledge)
The uncertainty in the model parameters $\theta$ due to lack of knowledge (finite data).
*   **Reducible:** Can be reduced by observing more data in the sparse region of the manifold.
*   **Indicator of OOD:** High epistemic uncertainty implies the input $x$ is far from the training distribution $P_{train}(x)$.
    *   *Example:* A model trained on European cities receiving an input from the Sahara Desert.

---

## 2. Modeling Aleatoric Uncertainty

We model the output not as a deterministic value but as a distribution (e.g., Gaussian for regression).

### 2.1 Heteroscedastic Regression Loss
The network predicts two heads: mean $\mu(x)$ and variance $\sigma^2(x)$.

The Negative Log Likelihood (NLL) for a Gaussian:

$$ \mathcal{L}_{NLL} = -\log P(y | \mu, \sigma) = \frac{1}{2\sigma^2} \| y - \mu \|^2 + \frac{1}{2} \log \sigma^2 + \text{const} $$

*   **Interpretation:**
    *   The first term is the MSE weighted by precision ($1/\sigma^2$). This allows the model to "attenuate" the loss on noisy samples by predicting high $\sigma^2$.
    *   The second term ($\log \sigma^2$) prevents the model from predicting infinite variance (collapse).

**Numerical Stability:** To avoid division by zero or negative variance, we usually predict $s = \log \sigma^2$ and optimize:
$$ \mathcal{L} = \frac{1}{2} \exp(-s) \| y - \mu \|^2 + \frac{1}{2} s $$

---

## 3. Modeling Epistemic Uncertainty (Bayesian DL)

Since exact Bayesian inference is intractable, we use Variational Inference (VI) or functional approximations.

### 3.1 Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016)

Dropout, usually viewed as regularization, can be interpreted as a variational Bayesian approximation.
*   **Training:** Dropout active.
*   **Inference:** Dropout **kept active**. We perform $T$ stochastic forward passes $\{\hat{y}_t\}_{t=1}^T$.

**Predictive Mean:**
$$ \mathbb{E}[y] \approx \frac{1}{T} \sum_{t=1}^T \hat{y}_t $$

**Predictive Entropy (Classification):**
$$ H(y|x) = - \sum_{c=1}^C p_{avg}(c) \log p_{avg}(c) $$
Where $p_{avg}$ is the numerical average of Softmax vectors.

**Predictive Variance (Regression):**
$$ \text{Var}(y) \approx \underbrace{\frac{1}{T} \sum \sigma^2_t}_{\text{Aleatoric}} + \underbrace{\frac{1}{T} \sum (\mu_t - \bar{\mu})^2}_{\text{Epistemic}} $$

### 3.2 Deep Ensembles (Lakshminarayanan et al., 2017)
Train $M$ models independently with random initialization and shuffled data.
*   **Pros:** Empirically the Gold Standard. Better calibration and OOD detection than MC Dropout.
*   **Cons:** $M \times$ training and inference cost (computationally heavy).
*   **Ununennium Strategy:** We support "Snapshots Ensembles" (Cyclic Learning Rate) to get ensemble diversity within a single training run.

### 3.3 Evidential Deep Learning (EDL) (Sensoy et al., 2018)

This deterministic method places a distribution over the distribution (Dirichlet Distribution for classification).
The network predicts evidence $e_k \ge 0$.
$$ \alpha_k = e_k + 1 $$
$$ S = \sum \alpha_k $$
$$ \text{Expected Probability } \hat{p}_k = \alpha_k / S $$
$$ \text{Uncertainty (Vacuity) } u = K / S $$

*   **Mechanism:** High evidence (lots of support) $\to$ High $S$ $\to$ Low $u$.
*   **Effect:** A pure OOD sample will have $e_k \approx 0$ for all classes, so $u \to 1$.
*   **Loss:** KL-divergence between predicted Dirichlet and Uniform Dirichlet + Accuracy term.

---

## 4. Calibration

A model is **Calibrated** if its predicted confidence matches its empirical accuracy.
$$ P(Y=y | \hat{P}=p) = p $$

### 4.1 Reliability Diagrams
Plot Accuracy vs Confidence in bins (e.g., 0-0.1, 0.1-0.2, ...).
*   **Perfect Calibration:** $y=x$ diagonal line.
*   **Under-confident:** Above diagonal.
*   **Over-confident:** Below diagonal (Typical DNN behavior).

### 4.2 Expected Calibration Error (ECE)
Weighted average gap.

$$ \text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} | \text{acc}(B_m) - \text{conf}(B_m) | $$

### 4.3 Temperature Scaling (Platt Scaling)
A post-processing step to fix calibration on a validation set.
$$ \hat{q}_i = \max_k \sigma_{SM}(\mathbf{z}_i / T)^{(k)} $$
We learn a single scalar parameter $T$ (Temperature) that softens ($T>1$) or sharpens ($T<1$) the logits $\mathbf{z}$ to minimize NLL.
*   **Note:** Does not change accuracy (rank preserving), only confidence values.

---

## 5. Conformal Prediction (CP)

Bayesian methods provide a *score* of uncertainty but no guarantees. **Conformal Prediction** provides rigorous statistical guarantees of finite-sample validity.

"We guarantee with probability $1-\alpha$ that the true label $y$ is contained in the predicted set $\mathcal{C}(x)$."

### 5.1 Inductive Conformal Prediction (ICP)

1.  **Calibration Set:** Hold out a set $\{(x_i, y_i)\}$.
2.  **Non-Conformity Score ($s_i$):** e.g., $s_i = 1 - \hat{P}(y_i | x_i)$. (How "weird" is the true label given the model?).
3.  **Quantile:** Compute $\hat{q}$ as the $\lceil (N+1)(1-\alpha) \rceil / N$ quantile of scores $s_i$.
4.  **Prediction:** For test $x_{new}$, include all classes $k$ where $1 - \hat{P}(k|x_{new}) \le \hat{q}$.

**Result:**
The model outputs a set of classes (e.g., `{"Forest", "Shrub"}`).
*   Easy sample $\to$ Set size 1.
*   Hard sample $\to$ Set size 5.
*   OOD sample $\to$ Set size $C$ (all classes).

Ununennium implements `ConformalClassifier` wrappers that transform any PyTorch model into a set predictor with coverage guarantees.

---

## 6. Applications in Earth Observation

### 6.1 Domain Adaptation
We use Epistemic Uncertainty heatmaps to detect when the satellite passes over a new biome.
$$ \text{if } \text{mean}(H(y)) > \tau \implies \text{Trigger Retraining/Labeling} $$

### 6.2 Reliable Mapping
Flood mapping requires high precision. We threshold the uncertainty map:
*   **Mask:** $y_{out} = \text{bg}$ if $u(x) > \tau$ else $\operatorname{argmax} P(y|x)$.
*   This creates "No Data" holes in the map where the model is unsure, rather than guessing.

### 6.3 Active Learning
The `ununennium.active_learning` module prioritizes labeling patches with high Epistemic Uncertainty (maximizing information gain).

---

## 7. Ununennium API

```python
from ununennium.models import DeepLabV3Plus
from ununennium.uncertainty import BayesianWrapper

# 1. Wrap model
model = DeepLabV3Plus(..., dropout_rate=0.2)
bayesian_model = BayesianWrapper(model, method="mc_dropout", n_samples=30)

# 2. Predict
mean, entropy, variance = bayesian_model(input_tensor)

# 3. Calibrate
calibrator = TemperatureScaling()
calibrator.fit(val_logits, val_labels)
calibrated_probs = calibrator.transform(test_logits)
```

---

## 8. References

1.  **Gal, Y., & Ghahramani, Z. (2016).** "Dropout as a bayesian approximation: Representing model uncertainty in deep learning". *ICML*.
2.  **Lakshminarayanan, B., et al. (2017).** "Simple and scalable predictive uncertainty estimation using deep ensembles". *NeurIPS*.
3.  **Kendall, A., & Gal, Y. (2017).** "What uncertainties do we need in bayesian deep learning for computer vision?". *NeurIPS*.
4.  **Sensoy, M., et al. (2018).** "Evidential deep learning to quantify classification uncertainty". *NeurIPS*.
5.  **Angelopoulos, A. N., & Bates, S. (2021).** "A gentle introduction to conformal prediction and distribution-free uncertainty quantification". *arXiv*.
