# Task 1 — Classical ML Baseline: Implementation Plan

**Objective**: Establish a rigorous classical ML baseline on the synthetic regime-switching dataset. This baseline is the apples-to-apples comparator for all quantum feature augmentation experiments.

---

## 1. Data Generation

### DGP Summary

```
Regime probabilities: P(R1) = 0.75, P(R2) = 0.25  (latent — not observed)

Regime 1 target:  Y = 2·X1 − X2 + ε
Regime 2 target:  Y = X1·X3 + log(|X2| + 1) + ε

ε ~ N(0, 1) in both regimes
```

### Feature Distributions

| Feature | Regime 1           | Regime 2                                           |
|---------|--------------------|----------------------------------------------------|
| X1      | N(0, 1)            | N(3, 1) — correlated with X3 (ρ = 0.8)            |
| X2      | N(0, 1)            | Cauchy(0, 1) — heavy tails, no finite variance     |
| X3      | N(0, 1)            | N(3, 1) — correlated with X1 (ρ = 0.8)            |
| X4      | Uniform(−1, 1)     | Exponential(λ=1) — pure noise in both regimes      |

### Dataset Sizes

```python
N_TRAIN = 10_000
N_TEST  = 10_000
RANDOM_SEEDS = [42, 123, 7, 999, 2026]  # run all five for variance estimates
```

### Generation Skeleton

```python
import numpy as np

def generate_regime_data(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X      : (n, 4)  raw features [X1, X2, X3, X4]
    Y      : (n,)    target
    regime : (n,)    latent regime labels (1 or 2) — store but do NOT pass to model
    """
    rng = np.random.default_rng(seed)

    regime = rng.choice([1, 2], size=n, p=[0.75, 0.25])
    mask1 = regime == 1
    mask2 = regime == 2
    n1, n2 = mask1.sum(), mask2.sum()

    X = np.zeros((n, 4))
    Y = np.zeros(n)

    # --- Regime 1 ---
    X[mask1, 0] = rng.standard_normal(n1)          # X1
    X[mask1, 1] = rng.standard_normal(n1)          # X2
    X[mask1, 2] = rng.standard_normal(n1)          # X3
    X[mask1, 3] = rng.uniform(-1, 1, n1)           # X4
    eps1 = rng.standard_normal(n1)
    Y[mask1] = 2 * X[mask1, 0] - X[mask1, 1] + eps1

    # --- Regime 2 ---
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    mu  = np.array([3.0, 3.0])
    x13 = rng.multivariate_normal(mu, cov, n2)
    X[mask2, 0] = x13[:, 0]                        # X1
    X[mask2, 1] = rng.standard_cauchy(n2)          # X2
    X[mask2, 2] = x13[:, 1]                        # X3
    X[mask2, 3] = rng.exponential(1.0, n2)         # X4
    eps2 = rng.standard_normal(n2)
    Y[mask2] = X[mask2, 0] * X[mask2, 2] + np.log(np.abs(X[mask2, 1]) + 1) + eps2

    return X, Y, regime
```

---

## 2. Preprocessing

### 2.1 Winsorization (before scaling)

Regime 2 draws X2 from a Cauchy distribution — infinite variance, extreme outliers that will destabilize polynomial features. Winsorize before all transforms.

```python
from scipy.stats import mstats

def winsorize(X: np.ndarray, limits=(0.01, 0.01)) -> np.ndarray:
    """Clip each feature to [1st, 99th] percentile, computed on TRAIN only."""
    return np.column_stack([
        mstats.winsorize(X[:, j], limits=limits) for j in range(X.shape[1])
    ])

# Fit clip thresholds on train, apply same bounds to test
X_train_w = winsorize(X_train)
X_test_w  = np.clip(X_test, X_train_w.min(axis=0), X_train_w.max(axis=0))
```

### 2.2 Standard Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train_w)
X_train_s = scaler.transform(X_train_w)
X_test_s  = scaler.transform(X_test_w)
```

> All transforms below operate on `X_train_s` / `X_test_s`.

---

## 3. Feature Augmentation — Transformations

The DGP for Regime 2 uses `X1·X3` and `log(|X2|+1)` directly. The classical feature set must include these. All other transforms are added for generality so the model can discover the structure without oracle knowledge.

### 3.1 Raw Features (always included)

```
X1, X2, X3, X4
```

### 3.2 Polynomial — Degree 2

All squared terms and pairwise interactions.

```python
from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
# Adds: X1², X2², X3², X4², X1·X2, X1·X3, X1·X4, X2·X3, X2·X4, X3·X4
#       (plus the raw features — PolynomialFeatures includes degree-1 by default)
```

Key interactions present in DGP: **X1·X3** (Regime 2 multiplicative signal).

### 3.3 Polynomial — Degree 3 (optional extension)

```python
poly3 = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
# Warning: 84 features from 4 inputs — needs strong regularization
```

Use only with Lasso or aggressive Ridge. Compare test MSE against degree-2 to check if extra complexity helps.

### 3.4 Log and Absolute Transforms

Directly targets Regime 2 signal `log(|X2| + 1)`.

```python
def log_abs_features(X: np.ndarray) -> np.ndarray:
    """
    Returns shape (n, 8): [|X1|, |X2|, |X3|, |X4|,
                            log(|X1|+1), log(|X2|+1), log(|X3|+1), log(|X4|+1)]
    """
    abs_X = np.abs(X)
    log_X = np.log(abs_X + 1)
    return np.hstack([abs_X, log_X])
```

### 3.5 Sign Features

Regime 2 Cauchy X2 has symmetric heavy tails — the sign can carry information about which side of the distribution the sample fell on.

```python
def sign_features(X: np.ndarray) -> np.ndarray:
    """Returns shape (n, 4): [sign(X1), sign(X2), sign(X3), sign(X4)]"""
    return np.sign(X)
```

### 3.6 Cross-Log Interactions (targeted)

Explicitly encode the nonlinear Regime 2 term as a feature.

```python
def cross_log_features(X: np.ndarray) -> np.ndarray:
    """
    Targeted features derived from DGP structure:
      - log(|X2|+1) * X1   (product of Regime 2 terms)
      - X1 * X3             (oracle Regime 2 interaction — also in poly2)
      - |X2| * X3
    Returns shape (n, 3)
    """
    log_abs_x2 = np.log(np.abs(X[:, 1]) + 1)
    return np.column_stack([
        log_abs_x2 * X[:, 0],
        X[:, 0] * X[:, 2],
        np.abs(X[:, 1]) * X[:, 2],
    ])
```

### 3.7 Full Feature Assembly

```python
def build_classical_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Assembles all classical features for a given X (already scaled).
    degree: polynomial degree (2 or 3)
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    
    X_poly      = poly.fit_transform(X)   # includes raw + poly terms
    X_log_abs   = log_abs_features(X)
    X_sign      = sign_features(X)
    X_cross_log = cross_log_features(X)
    
    return np.hstack([X_poly, X_log_abs, X_sign, X_cross_log])
```

> **Fit poly on train only. Apply the fitted transformer to test.**
> `poly.fit(X_train_s)` then `poly.transform(X_test_s)` — never `fit_transform` on test.

---

## 4. Feature Matrix Summary

| Block                    | # Features (deg=2) | Key signal captured                |
|--------------------------|--------------------|------------------------------------|
| Raw (via PolynomialFeatures) | 4              | Linear Regime 1 signal             |
| Degree-2 poly            | 10                 | X1·X3 (Regime 2), pairwise linear  |
| Abs values               | 4                  | Regime shift in X2 magnitude       |
| Log(abs + 1)             | 4                  | **log(\|X2\|+1)** (Regime 2 exact) |
| Sign                     | 4                  | Heavy-tail asymmetry of Cauchy X2  |
| Cross-log interactions   | 3                  | Composed Regime 2 features         |
| **Total**                | **29**             |                                    |

With degree=3: expands to ~113 features — regularization becomes critical.

---

## 5. Models and Regularization

The same model class must be used for classical and quantum baselines (per challenge spec).

### 5.1 Model Ladder

Run all four. Compare using out-of-sample MSE, MAE, Pearson correlation.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

models = {
    "LR_raw":        LinearRegression(),
    "LR_classical":  LinearRegression(),        # full feature set, unregularized
    "Ridge_classical": Ridge(alpha=1.0),        # alpha tuned via CV
    "Lasso_classical": Lasso(alpha=0.01),       # alpha tuned via CV
}
```

### 5.2 Alpha Selection

Use 5-fold CV on the training set. Never touch test data during tuning.

```python
from sklearn.model_selection import GridSearchCV

alpha_grid = {"alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]}

ridge_cv = GridSearchCV(Ridge(), alpha_grid, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_aug, y_train)
best_alpha_ridge = ridge_cv.best_params_["alpha"]

lasso_cv = GridSearchCV(Lasso(max_iter=10_000), alpha_grid, cv=5, scoring="neg_mean_squared_error")
lasso_cv.fit(X_train_aug, y_train)
best_alpha_lasso = lasso_cv.best_params_["alpha"]
```

### 5.3 Optional: Non-Linear Extension

After establishing the linear baseline, add a non-linear model on raw features only to disentangle feature effect from model complexity.

```python
from sklearn.ensemble import GradientBoostingRegressor

# Raw features only — isolates model complexity, not feature engineering
gbr_raw = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
```

Label these results clearly as "model complexity extension" — they cannot be compared directly to the linear models.

---

## 6. Evaluation

### 6.1 Metrics

```python
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(y_true, y_pred, label=""):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {"model": label, "MSE": mse, "MAE": mae, "Pearson_r": corr}
```

### 6.2 Per-Regime Evaluation

Since we stored the latent regime label at generation time, evaluate separately on Regime 1 and Regime 2 test subsets. This reveals whether a model is capturing both regimes or gaming the aggregate metric.

```python
mask_r1_test = (regime_test == 1)
mask_r2_test = (regime_test == 2)

eval_r1 = evaluate(y_test[mask_r1_test], y_pred[mask_r1_test], label + "_R1")
eval_r2 = evaluate(y_test[mask_r2_test], y_pred[mask_r2_test], label + "_R2")
```

### 6.3 Multi-Seed Stability

Run every experiment across all 5 seeds. Report mean ± std. A model that wins on a single seed but has high variance is not robust.

```python
seed_results = []
for seed in RANDOM_SEEDS:
    X_tr, y_tr, r_tr = generate_regime_data(N_TRAIN, seed)
    X_te, y_te, r_te = generate_regime_data(N_TEST,  seed + 1000)
    # ... preprocess, fit, evaluate ...
    seed_results.append(metrics)

import pandas as pd
df = pd.DataFrame(seed_results)
print(df.groupby("model")[["MSE", "MAE", "Pearson_r"]].agg(["mean", "std"]))
```

### 6.4 Results Table Template

| Model                  | MSE (mean ± std) | MAE (mean ± std) | Pearson r (mean ± std) |
|------------------------|------------------|------------------|------------------------|
| LR — raw features      |                  |                  |                        |
| LR — classical aug     |                  |                  |                        |
| Ridge — classical aug  |                  |                  |                        |
| Lasso — classical aug  |                  |                  |                        |
| *(quantum aug — TBD)*  |                  |                  |                        |

---

## 7. Overfitting Diagnostics

Report these alongside test metrics for every model:

- **Train vs test MSE gap**: gap > 0.2× train MSE is a warning sign.
- **Lasso sparsity**: how many coefficients are zeroed out? Which features survive?
- **Learning curves**: MSE vs. training set size (subsample train at 20%, 40%, 60%, 80%, 100%).
- **Feature importance**: Ridge coefficients scaled by feature std — reveals which transforms carry the signal.

```python
import matplotlib.pyplot as plt

def plot_coeff_importance(model, feature_names, title=""):
    coef = model.coef_
    std  = X_train_aug.std(axis=0)
    importance = np.abs(coef * std)
    idx = np.argsort(importance)[::-1][:20]  # top 20
    plt.barh(range(len(idx)), importance[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.title(title)
    plt.tight_layout()
```

---

## 8. Baseline Deliverables

Before moving to quantum features, the following must exist:

1. `generate_regime_data()` tested and reproducible across seeds.
2. `build_classical_features()` with documented feature count and names.
3. Tuned Ridge and Lasso alphas (stored, not re-tuned on quantum runs).
4. Results table filled for all 4 linear models, all 5 seeds.
5. Per-regime breakdown for the best classical model.
6. Coefficient importance plot for Ridge/Lasso.

These are the fixed reference numbers that all quantum experiments compare against.

---

## 9. Implementation Order

```
[ ] 1. generate_regime_data() + sanity checks (distribution plots, regime balance)
[ ] 2. Winsorize + StandardScaler pipeline
[ ] 3. build_classical_features() — confirm output shape = (n, 29)
[ ] 4. LR on raw features — establish floor
[ ] 5. LR on classical aug features
[ ] 6. Ridge CV alpha search
[ ] 7. Lasso CV alpha search
[ ] 8. Multi-seed loop (5 seeds)
[ ] 9. Per-regime evaluation
[ ] 10. Overfitting diagnostics + coefficient plot
[ ] 11. Freeze results table — handoff to quantum experiments
```
