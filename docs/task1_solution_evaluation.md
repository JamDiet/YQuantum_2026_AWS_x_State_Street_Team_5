# Task 1 — Solution Evaluation & Quantum Feature Analysis

Evaluation of `QFA_Solution.ipynb` against `task1_classical_baseline_plan.md`.

---

## 1. What the Solution Implements (Summary)

**Part I (Synthetic)**
- Data: `generate_regime_data(n=10000, seed=42)` — single seed
- Preprocessing: `StandardScaler` → clip at ±5σ → `π·tanh(x/2)` squash for quantum inputs
- Classical: LR (raw), Ridge poly-deg2 (α=1.0), Ridge poly-deg3 (α=1.0), Lasso poly-deg2 (α=0.01)
- Quantum: 3 circuits (Angle, ZZ, IQP), each producing 10 features (4 single-Z + 6 pairwise-ZZ)
- Quantum augmented: Ridge (raw + Q), Ridge (poly2 + Q), all at hardcoded α=1.0
- Best result: Ridge poly-deg3 MSE=1.220, Corr=0.909

**Part II (Stock)**
- 10 S&P 500 tickers via yfinance (fallback: synthetic GBM); 15 stock-minus-market features
- 15-qubit Angle circuit → 120 quantum features
- Walk-forward: 504-day train window, 5-day roll step
- Ridge (α=1.0) for classical, quantum-aug, and poly2+quantum-aug

---

## 2. Evaluation Against Plan

### 2.1 Data Generation — MATCHES

Solution matches the DGP spec exactly:
- Regime probabilities 75/25 ✓
- Cauchy X2 in Regime 2 — **clipped at [-10, 10]** rather than raw Cauchy ✓ (sensible choice)
- Multivariate normal X1/X3 with ρ=0.8, mean 3 ✓
- Y formulas correct ✓
- Regime labels stored as `r_train`, `r_test` ✓

### 2.2 Preprocessing — PARTIALLY MATCHES, ONE DIFFERENCE

| Step | Plan | Solution | Verdict |
|------|------|----------|---------|
| Cauchy tail handling | `scipy.mstats.winsorize` at 1%/99% (percentile-based) | Clip at ±5σ after scaling | Equivalent in intent, different mechanically |
| StandardScaler | Fit on train, apply to test | Fit on train, apply to test ✓ | Matches |
| Quantum squash | Not in plan | `π·tanh(x/2)` to [-π, π] | **Addition not in plan** |

**Note on ±5σ clip vs percentile winsorize**: For Cauchy-distributed X2, ±5σ after standardization is not equivalent to 1%/99% percentile — Cauchy has undefined variance so "5σ" is ambiguous. The StandardScaler will fit very badly on Cauchy draws before clipping. The solution clips *after* scaling, which means the scaler's mean/std are computed on extreme-outlier-contaminated data first. Percentile-based winsorize *before* scaling (as in the plan) is more robust. Flag this.

**Note on quantum squash**: `π·tanh(x/2)` is a smooth sigmoid-like map to (-π, π). This is appropriate for angle encoding — RY(θ) is 2π-periodic, so bounding inputs avoids aliasing. **Classical features and quantum features are now derived from differently preprocessed inputs** (`X_train_s` vs `X_train_q`). This is not wrong, but it means classical and quantum pipelines diverge at preprocessing, which complicates the "apples-to-apples" comparison. Document this explicitly in results.

### 2.3 Classical Feature Augmentation — PARTIAL GAP

| Transform | Plan | Solution | Verdict |
|-----------|------|----------|---------|
| PolynomialFeatures deg=2 | ✓ (14 features) | ✓ (14 features) | Matches |
| PolynomialFeatures deg=3 | Optional extension | ✓ implemented (34 features) | Solution goes further — good |
| `log(\|Xi\|+1)` | ✓ explicitly included | **Absent** | **Gap** |
| `\|Xi\|` | ✓ included | **Absent** | **Gap** |
| Sign features | ✓ included | **Absent** | **Gap** |
| Cross-log interactions | ✓ (targeted DGP terms) | **Absent** | **Gap** |

**This is the most important gap.** The DGP for Regime 2 is `Y = X1·X3 + log(|X2|+1) + ε`. The `log(|X2|+1)` term is directly computable as a classical feature and is absent from the solution's classical baseline. This means the classical baseline in the solution is weaker than it could be, which inflates the apparent value of quantum features. Before concluding quantum augmentation helps, the log/abs features should be added to the classical baseline and results re-evaluated.

> **Action**: Add `log_abs_features()` and `cross_log_features()` to the classical pipeline per the plan. This is the real apples-to-apples floor.

### 2.4 Models and Regularization — MOSTLY MATCHES, ALPHA TUNING ABSENT

| Aspect | Plan | Solution | Verdict |
|--------|------|----------|---------|
| LR raw features | ✓ | ✓ | Matches |
| LR poly-deg2 | ✓ | ✓ (Ridge not LR — effectively same) | Minor diff |
| Ridge poly-deg2 | ✓ α=CV-tuned | ✓ α=1.0 **hardcoded** | **No CV tuning** |
| Ridge poly-deg3 | Optional | ✓ α=1.0 hardcoded | Good addition |
| Lasso poly-deg2 | ✓ α=CV-tuned | ✓ α=0.01 hardcoded | **No CV tuning** |
| Same model class rule | ✓ requirement noted | ✓ Ridge used throughout | Matches |

**Alpha hardcoding is a problem for fairness.** `Ridge(alpha=1.0)` is applied to:
- 4 features (raw)
- 14 features (poly2)
- 34 features (poly3)
- 24 features (raw + 10 quantum)
- 24 features (poly2 + 10 quantum)
- 135 features (poly2 + 120 quantum, Part II)

The optimal regularization strength scales with feature count and feature covariance. Applying α=1.0 uniformly treats a 4-feature model identically to a 135-feature model, which is not rigorous. CV alpha selection should be used, or at minimum the sensitivity to alpha should be reported.

### 2.5 Evaluation — GAP IN MULTI-SEED AND PER-REGIME

| Aspect | Plan | Solution | Verdict |
|--------|------|----------|---------|
| MSE, MAE, Pearson r | ✓ | ✓ | Matches |
| Per-regime breakdown | ✓ using stored r_test | **Not performed** | **Gap** |
| Multi-seed (5 seeds) | ✓ | **Single seed=42** | **Gap** |
| Overfitting diagnostics | ✓ train/test gap | **Not explicitly reported** | Gap |
| Feature coefficient plot | ✓ | **Not present** | Gap |

The solution stores `r_train` and `r_test` but never uses them to break down performance per regime. This matters: a model could score well on aggregate MSE by fitting Regime 1 (75% of data) while completely failing on Regime 2. The per-regime breakdown would reveal this and is directly relevant to the quantum question (quantum circuits may preferentially capture nonlinear Regime 2 structure).

Single-seed results are also fragile — a ±0.05 swing in Pearson correlation across seeds could change the conclusion about quantum feature value.

---

## 3. Watch-Outs and Potential Issues

### 3.1 Cauchy Scaler Contamination (Medium Risk)

```python
# Solution order:
scaler = StandardScaler()
X_train_s = np.clip(scaler.fit_transform(X_train), -5, 5)
```

`StandardScaler` computes mean and std over raw `X_train`, which includes Cauchy X2 draws. Cauchy has undefined mean and variance — in practice a 10k sample will have extreme values that pull the estimated std upward, compressing all other features. Clipping *after* scaling with a contaminated scaler is less effective than clipping first. The correct order: winsorize/clip raw data → fit StandardScaler on clipped data → transform.

### 3.2 Zero-Weight Entanglement in Angle Circuit (Low–Medium Risk)

```python
weights = np.zeros((2, 4))
qml.BasicEntanglerLayers(weights, wires=range(4), rotation=qml.RY)
```

`BasicEntanglerLayers` alternates CNOT rings with parameterized single-qubit rotations. With `weights=np.zeros`, the RY rotations are all RY(0) = Identity. The circuit reduces to: `AngleEmbedding → CNOT_ring → CNOT_ring`. The entangling structure is still present (CNOT creates qubit correlations) but the variational rotations contribute nothing. The pairwise ⟨ZZ⟩ features therefore capture only CNOT-mediated correlations between the angle-embedded qubits — a specific and relatively simple structure.

This is not wrong as a fixed feature map, but the description implies learnable parameters. If the intent is fixed feature extraction, the zero-weight circuit is a valid choice and should be documented as such.

### 3.3 Preprocessing Asymmetry Between Classical and Quantum Inputs (Medium Risk)

Classical models receive `X_train_s` (StandardScaler output, clipped ±5).  
Quantum circuits receive `X_train_q = π·tanh(X_train_s/2)` (additionally squashed).

When the augmented matrix `np.hstack([X_train_s, Q_tr])` is built, the classical columns are in a different scale/distribution than the quantum columns. Ridge regression is scale-sensitive through the L2 penalty — the regularization will penalize classical and quantum feature coefficients differently even at the same α. A secondary StandardScaler over the combined augmented matrix before fitting Ridge would make the comparison cleaner.

### 3.4 Part II: High Feature-to-Sample Ratio in Walk-Forward (Medium Risk)

In Part II, `Ridge(alpha=1.0)` is fitted on:
- `np.hstack([Xp2_tr, Qtr])` = (504, 135 + 120) = (504, 255) matrix

255 features on 504 samples is a 1:2 ratio. With α=1.0 this is probably well-regularized, but the optimal α has not been verified. If α is too small, models could overfit within the 504-day window and IC estimates would be optimistic. The results (mean IC +0.106 for Q+Poly2) should be treated with caution until alpha sensitivity is confirmed.

### 3.5 Part II yfinance Fallback Synthetic Data (Low Risk, High Transparency Impact)

```python
USE_REAL_DATA = True  # Toggle
```

If `yfinance` is unavailable, the notebook falls back to a custom GBM regime-switching simulator with hardcoded correlation parameters. Results generated with synthetic stock data would not be comparable to real-data results and would require separate reporting. Confirm which data source was used when results are quoted.

### 3.6 Poly3 Beats Quantum — The Right Conclusion? (Interpretive)

The solution's headline result: poly-deg3 (MSE=1.220) outperforms all quantum-augmented poly-deg2 models. This is presented as a valid negative result. However:

- Poly-deg3 is a 34-feature model; Quantum+Poly2 is a 24-feature model — not the same complexity tier
- The log/abs classical features (directly encoding the Regime 2 DGP) are absent from the classical baseline
- A fair comparison would be: [Quantum + Poly2] vs [Poly2 + log/abs/sign] on the same alpha

Until the log/abs features are added, it is premature to conclude that quantum features provide incremental value over the best classical baseline.

---

## 4. Quantum Feature Transformations — Mathematical Basis

### 4.1 Preprocessing for Quantum Input

All quantum circuits in the solution use:

$$x_i^{(q)} = \pi \cdot \tanh\!\left(\frac{x_i^{(s)}}{2}\right)$$

where $x_i^{(s)}$ is the standardized input. This maps $\mathbb{R} \to (-\pi, \pi)$. The motivation: RY(θ) is $2\pi$-periodic, so large inputs would wrap around and create aliasing artifacts. The tanh squash ensures the encoding is injective over the input range. It also compresses the heavy Cauchy tails more aggressively than ±5σ clipping does.

### 4.2 Angle Encoding (AngleEmbedding)

**Circuit:**
$$U_{\text{angle}}(\mathbf{x}) = \text{CNOT\_ring}^2 \cdot \bigotimes_{i=1}^{4} R_Y(x_i^{(q)})$$

**State after encoding qubit $i$:**
$$R_Y(x_i)|0\rangle = \cos\!\left(\frac{x_i}{2}\right)|0\rangle + \sin\!\left(\frac{x_i}{2}\right)|1\rangle$$

**Single-qubit expectation (no entanglement):**
$$\langle Z_i \rangle = \cos(x_i^{(q)})$$

After the CNOT ring, the two-qubit expectations $\langle Z_i Z_j \rangle$ are more complex products of cosine/sine terms. With zero-weight RY layers the CNOT structure is fixed, giving:

$$\phi_{\text{angle}}(\mathbf{x}) \approx \left[\cos(x_i^{(q)}),\ \cos(x_i^{(q)})\cos(x_j^{(q)}) \pm \sin(x_i^{(q)})\sin(x_j^{(q)})\right]_{i,j}$$

**Classical analogy**: This is similar to a Fourier feature map — the features are trigonometric functions of the inputs rather than polynomials. The relevant identity:

$$\cos(x_i)\cos(x_j) = \frac{1}{2}[\cos(x_i - x_j) + \cos(x_i + x_j)]$$

So pairwise ZZ measurements capture sum/difference frequency components, which are distinct from polynomial $X_i X_j$ interactions. This is the feature space the angle circuit is exploring.

**Push-off point**: To test whether these frequency-space features add anything over the polynomial basis, examine the Lasso coefficients on the augmented `[poly2, angle_features]` matrix. If Lasso zeros out the cosine features while retaining polynomial features, the angle encoding is redundant for this DGP.

### 4.3 ZZ Feature Map (Havlíček et al., Nature 2019)

**Circuit (1 rep shown):**
$$U_{\text{ZZ}}(\mathbf{x}) = \left(\prod_{i<j} e^{-i(\pi - x_i)(\pi - x_j) Z_i Z_j / 2}\right) \left(\prod_i R_Z(x_i)\right) H^{\otimes n}$$

**Key entangling gate:**
$$\text{CNOT} \cdot R_Z\!\left((\pi - x_i)(\pi - x_j)\right) \cdot \text{CNOT} = e^{-i \frac{(\pi-x_i)(\pi-x_j)}{2} Z_i Z_j}$$

This implements a ZZ interaction where the rotation angle encodes the **product** of two features:

$$\phi_{ij}(\mathbf{x}) = (\pi - x_i^{(q)})(\pi - x_j^{(q)})$$

Expanding: $= \pi^2 - \pi(x_i + x_j) + x_i x_j$

The product $x_i x_j$ is the Regime 2 signal $X_1 X_3$. The ZZ circuit is the only one that explicitly encodes pairwise products in its gate angles, giving it a structural reason to capture the Regime 2 multiplicative interaction.

**Quantum kernel induced by this map:**
$$k_{\text{ZZ}}(\mathbf{x}, \mathbf{x}') = |\langle 0|U_{\text{ZZ}}^\dagger(\mathbf{x}') U_{\text{ZZ}}(\mathbf{x})|0\rangle|^2$$

This kernel is believed to be classically hard to compute exactly for sufficient depth/width (Havlíček et al. provide hardness arguments). The 2-rep version in the solution is shallow enough that classical simulation is fast, but the feature space structure is still distinct from polynomial kernels.

**Push-off point**: ZZ features most directly capture $X_1 X_3$ through the $(\pi - x_i)(\pi - x_j)$ encoding. Compare ZZ performance specifically on the Regime 2 test subset (using stored `r_test`) — this is where ZZ should show the largest advantage over angle and IQP encodings.

### 4.4 IQP Encoding (Shepherd & Bremner 2009; Havlíček 2019 application)

**Structure:** Alternating Hadamard layers and diagonal phase gates:
$$U_{\text{IQP}}(\mathbf{x}) = H^{\otimes n} \cdot D(\mathbf{x}) \cdot H^{\otimes n}$$

where $D(\mathbf{x}) = \bigotimes_i e^{i x_i Z_i} \bigotimes_{i<j} e^{i x_i x_j Z_i Z_j}$ is diagonal in the computational basis.

PennyLane's `IQPEmbedding` with `n_repeats=2` applies this structure twice:
$$U_{\text{IQP}}^{(2)} = (H^{\otimes n} D(\mathbf{x}))^2 H^{\otimes n}$$

**Key property**: IQP circuits are a subset of circuits where all gates commute (they are diagonal in one basis). Sampling from IQP output distributions is believed to be classically hard even for polynomial-time classical algorithms (under plausible complexity assumptions). However, we are not sampling — we are computing expectation values, which can be classically simulated efficiently for shallow circuits.

**Feature space**: The expectation values $\langle Z_i \rangle$ and $\langle Z_i Z_j \rangle$ under IQP encoding are related to the Fourier coefficients of the Boolean function defined by $D(\mathbf{x})$. This is a different functional basis than both polynomial (ZZ) and trigonometric (Angle) feature spaces.

**Push-off point**: IQP has depth=1 (shallowest) but is reported to perform comparably to ZZ and Angle in the solution. This is worth investigating — if a very shallow circuit matches deeper ones, either (a) the extra depth adds little, or (b) all three circuits are capturing essentially the same information. Compute pairwise correlation between angle, ZZ, and IQP feature matrices to check redundancy.

### 4.5 Feature Space Comparison Summary

| Circuit | Feature Type | Captures $X_1 X_3$? | Captures $\log\|X_2\|$? | Depth | Features |
|---------|-------------|---------------------|--------------------------|-------|----------|
| Angle | Trigonometric (cosine basis) | Indirectly via $\cos(x_i)\cos(x_j)$ | No | 2 | 10 |
| ZZ | Polynomial-phase products | **Yes** — $(π-x_i)(π-x_j) \supset x_i x_j$ | No | 31 | 10 |
| IQP | Fourier/Boolean coefficients | Indirectly | No | 1 | 10 |
| Poly-deg2 (classical) | Polynomial $x_i x_j$ | **Yes** — directly | No | — | 14 |
| Log/abs (plan, absent) | Nonlinear monotone | No | **Yes** — directly | — | 8 |

**Critical observation**: None of the three quantum circuits directly capture $\log|X_2|$, which is a primary Regime 2 signal. Poly-deg3 gets closer through high-order polynomial approximations of log (Taylor series), but log/abs features do it exactly. This means:

- Quantum circuits are competing against an incomplete classical baseline
- The `log(|X2|+1)` signal is potentially unclaimed by quantum features
- Adding log/abs to classical will raise the bar and force a cleaner comparison

### 4.6 Mathematical Basis for Pushing Further

**Kernel perspective**: Each quantum feature map implicitly defines a kernel:
$$k_q(\mathbf{x}, \mathbf{x}') = \phi_q(\mathbf{x})^T \phi_q(\mathbf{x}')$$

For angle encoding this gives:
$$k_{\text{angle}}(\mathbf{x}, \mathbf{x}') = \sum_i \cos(x_i)\cos(x_i') + \sum_{i<j} \langle Z_i Z_j \rangle(\mathbf{x}) \langle Z_i Z_j \rangle(\mathbf{x}')$$

This is a **low-rank** approximation of the full quantum kernel (which would use the full state overlap). The 10-feature extraction is analogous to random kitchen sinks — a Monte Carlo approximation of the full kernel.

**Richer extraction directions**:
1. **Probability features**: Extract all $2^4 = 16$ computational basis probabilities instead of 10 expectation values. Richer but risks overfitting with 16k features on 10k samples.
2. **Pauli-X/Y observables**: $\langle X_i \rangle$, $\langle Y_i \rangle$ add 8 more features per circuit — different functional dependencies on the input angles.
3. **More qubits than features**: Map 4 features into 8 qubits with repeated angle encodings — gives exponentially larger Hilbert space for the same 4 inputs.
4. **VQC with trained weights**: Replace `np.zeros((2, 4))` with learned weights. Optimize weights to maximize correlation between circuit output and Y on training set (proxy objective). This converts fixed feature extraction into adaptive feature learning.

---

## 5. Summary Assessment

| Category | Status | Priority |
|----------|--------|----------|
| DGP implementation | Correct | — |
| Preprocessing order (Cauchy scaler) | Risky | Fix before final results |
| Classical baseline completeness (log/abs missing) | Gap | **High — add before quantum comparison** |
| Alpha tuning | Not done | Medium — at minimum report sensitivity |
| Per-regime evaluation | Missing | High — core to the scientific question |
| Multi-seed stability | Missing | Medium — add 2–3 seeds |
| Preprocessing asymmetry (classical vs quantum inputs) | Present | Document explicitly |
| Zero-weight entanglement layers | Benign but document | Low |
| Part II feature/sample ratio | Flag | Medium |
| Quantum feature mathematical grounding | Sound, extensions documented above | — |
| Overall conclusion validity | Conditional — needs log/abs baseline first | — |
