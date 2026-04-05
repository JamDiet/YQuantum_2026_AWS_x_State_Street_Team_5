# QFA Mathematical Reference
## YQuantum 2026 — AWS × State Street Challenge

This document explains every mathematical operation in the solution from first principles, with worked examples.

---

## Table of Contents

1. [The Data Generating Process](#1-the-data-generating-process)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Classical Feature Engineering](#3-classical-feature-engineering)
4. [Quantum Feature Maps](#4-quantum-feature-maps)
5. [Ridge & Lasso Regression](#5-ridge--lasso-regression)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Walk-Forward Backtest](#7-walk-forward-backtest)
8. [Why Quantum Features Are Different](#8-why-quantum-features-are-different)

---

## 1. The Data Generating Process

### 1.1 Regime Switching Model

The target Y is drawn from one of two regimes, chosen randomly at each observation:

```
P(Regime 1) = 0.75,   P(Regime 2) = 0.25
```

**Regime 1** (linear, 75% of data):
```
Y = 2·X₁ - X₂ + ε,    ε ~ N(0,1)
```

**Regime 2** (nonlinear, 25% of data):
```
Y = X₁·X₃ + log(|X₂| + 1) + ε,    ε ~ N(0,1)
```

The regime indicator is **latent** — the model never sees which regime a sample came from. It must learn to handle both from X alone.

### 1.2 Feature Distributions

| Feature | Regime 1 | Regime 2 | Role |
|---|---|---|---|
| X₁ | N(0,1) | N(3,1), corr 0.8 with X₃ | Linear signal in R1, interaction in R2 |
| X₂ | N(0,1) | Cauchy(0,1) — heavy tails | Linear in R1, log-nonlinear in R2 |
| X₃ | N(0,1) | N(3,1), corr 0.8 with X₁ | Irrelevant in R1, interaction in R2 |
| X₄ | Uniform(-1,1) | Exponential(1) | Pure noise in both regimes |

### 1.3 Worked Example — Sample Generation

Say we draw `regime=2`, then:
```
(X₁, X₃) ~ N([3,3], [[1, 0.8],[0.8, 1]])
         → e.g. X₁=3.4, X₃=3.1  (correlated, shifted to mean=3)

X₂ ~ Cauchy(0,1) → e.g. X₂ = 0.7  (or could be X₂ = 47.3 — extreme!)

Y = 3.4 × 3.1 + log(|0.7| + 1) + ε
  = 10.54 + log(1.7) + ε
  = 10.54 + 0.531 + ε
  = 11.07 + ε
```

**Why does this matter for feature engineering?**
- A linear model sees X₁=3.4, X₂=0.7, X₃=3.1 and must somehow recover the product `X₁·X₃`
- Without the interaction term `X₁·X₃`, or the transform `log(|X₂|+1)`, no linear model can fit Regime 2

---

## 2. Preprocessing Pipeline

### 2.1 Why Cauchy is Dangerous for StandardScaler

The Cauchy distribution has **no finite mean or variance** — they are mathematically undefined. `StandardScaler` computes:
```
μ = (1/n) Σ xᵢ        ← diverges as n → ∞ for Cauchy
σ = sqrt((1/n) Σ (xᵢ-μ)²)  ← also diverges
```

If you fit `StandardScaler` on raw Cauchy draws, a single extreme value like X₂=1000 can shift the estimated mean by hundreds and inflate σ massively, making all other features appear near zero.

**Wrong order (current code):**
```python
X_train_s = np.clip(scaler.fit_transform(X_train), -5, 5)
# Problem: StandardScaler sees the Cauchy extremes first
```

**Correct order:**
```python
X_train_clipped = np.clip(X_train, np.percentile(X_train, 1, axis=0),
                                    np.percentile(X_train, 99, axis=0))
X_train_s = scaler.fit_transform(X_train_clipped)
X_train_s = np.clip(X_train_s, -5, 5)  # second safety clip
```

### 2.2 StandardScaler

For each feature j across n samples:
```
μⱼ = (1/n) Σᵢ Xᵢⱼ
σⱼ = sqrt((1/n) Σᵢ (Xᵢⱼ - μⱼ)²)
X̃ᵢⱼ = (Xᵢⱼ - μⱼ) / σⱼ
```

**Example:** If X₁ has mean=0.02, std=1.03:
```
X₁ = 2.5  →  X̃₁ = (2.5 - 0.02) / 1.03 = 2.41
X₁ = -1.0 →  X̃₁ = (-1.0 - 0.02) / 1.03 = -0.99
```

**Critical rule:** Fit the scaler on training data only. Transform test data using training statistics. Never `fit_transform(X_test)` — this is data leakage.

### 2.3 Tanh Squashing for Quantum Encoding

Quantum rotation gates (Ry, Rz) take angles as input. The output of the gate is periodic in the angle with period 2π. To prevent the circuit from "wrapping around" multiple times (which would make different input values produce identical quantum states), we squash features to (-π, π):

```
x_quantum = π · tanh(x_scaled / 2)
```

**Why tanh?** It's a smooth, monotonic function that maps ℝ → (-1,1). Scaling by π gives us (-π, π).

**Example:**
```
x_scaled = 0.0   →  π·tanh(0)   = 0.000
x_scaled = 1.0   →  π·tanh(0.5) = 0.785·π ≈ 2.47
x_scaled = 3.0   →  π·tanh(1.5) = 0.905·π ≈ 2.84
x_scaled = 5.0   →  π·tanh(2.5) = 0.987·π ≈ 3.10  ← near π but not equal
x_scaled = 100.0 →  π·tanh(50)  ≈ π − ε          ← extreme, safely bounded
```

The clipping at ±5σ already constrains most values; tanh provides a smooth final bound.

---

## 3. Classical Feature Engineering

### 3.1 Polynomial Features (Degree 2)

For input vector **x** = (X₁, X₂, X₃, X₄), degree-2 polynomial expansion adds:

**Original features (degree 1):** X₁, X₂, X₃, X₄  (4 terms)

**Squared terms (degree 2):** X₁², X₂², X₃², X₄²  (4 terms)

**Cross-product terms:** X₁X₂, X₁X₃, X₁X₄, X₂X₃, X₂X₄, X₃X₄  (6 terms)

**Total: 14 features**

The key term is **X₁X₃** — this directly captures the Regime 2 interaction. A linear model with degree-2 features can learn the coefficient on X₁X₃ and fit Regime 2 exactly (up to noise).

**Example:**
```
x = (2.0, 0.5, 3.0, -0.3)
Poly2(x) = [2.0, 0.5, 3.0, -0.3,       ← original
            4.0, 0.25, 9.0, 0.09,       ← squared
            1.0, 6.0, -0.6, 1.5, -0.15, -0.9]  ← cross products
```

### 3.2 Polynomial Features (Degree 3)

Degree-3 adds cubic terms and mixed degree-3 terms:
- Cubics: X₁³, X₂³, X₃³, X₄³  (4 terms)
- Mixed: X₁²X₂, X₁²X₃, X₁X₂², X₁X₂X₃, ...  (many terms)

**Total: 34 features for 4 inputs**

The extra terms help because the Cauchy-distributed X₂ in Regime 2 introduces nonlinear relationships that benefit from higher-order polynomial structure.

### 3.3 Missing: Log/Abs Features (Critical Gap)

The Regime 2 DGP contains `log(|X₂| + 1)` directly. **No polynomial** (of any degree) can exactly represent a log function. The polynomial approximation to log(x) around x=1 is:

```
log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - ...
```

This converges slowly and requires very high degree. The correct approach is to add explicit log-transformed features:

```python
# Features the classical baseline SHOULD have but DOESN'T:
log_abs_X1 = log(|X₁| + 1)
log_abs_X2 = log(|X₂| + 1)   ← directly in Regime 2 DGP
log_abs_X3 = log(|X₃| + 1)
abs_X1     = |X₁|
abs_X2     = |X₂|
```

**Without these, the classical bar is unfairly low.** Any quantum improvement over the current classical baseline may just reflect this missing feature, not a genuine quantum advantage.

---

## 4. Quantum Feature Maps

### 4.1 What is a Qubit?

A qubit is a two-dimensional quantum state:
```
|ψ⟩ = α|0⟩ + β|1⟩,    where |α|² + |β|² = 1
```

Think of it as a point on the unit sphere (Bloch sphere). The north pole is |0⟩, south pole is |1⟩, and every other point is a superposition.

### 4.2 Rotation Gates

**Ry(θ) gate** — rotates the qubit around the Y-axis by angle θ:
```
Ry(θ) = [[cos(θ/2),  -sin(θ/2)],
          [sin(θ/2),   cos(θ/2)]]
```

Starting from |0⟩:
```
Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
```

**Example:** encoding X₁ = 0.5 (after tanh squashing, x_q ≈ 0.785):
```
Ry(0.785)|0⟩ = cos(0.393)|0⟩ + sin(0.393)|1⟩
             = 0.924|0⟩ + 0.383|1⟩
```

**Rz(θ) gate** — rotates around the Z-axis:
```
Rz(θ) = [[e^(-iθ/2),        0   ],
          [0,          e^(iθ/2)  ]]
```

### 4.3 Expectation Values as Features

After encoding data into a quantum state |ψ(x)⟩, we extract features by measuring observables.

**Single-qubit Pauli-Z measurement:**
```
⟨Z_k⟩ = ⟨ψ(x)|Z_k|ψ(x)⟩ ∈ [-1, 1]
```

For a qubit in state α|0⟩ + β|1⟩:
```
⟨Z⟩ = |α|² - |β|² = cos²(θ/2) - sin²(θ/2) = cos(θ)
```

So `⟨Z_k⟩ = cos(x_k)` for angle encoding. This is a **nonlinear** function of the input.

**Two-qubit ZZ measurement:**
```
⟨Z_j ⊗ Z_k⟩ = ⟨ψ(x)|Z_j Z_k|ψ(x)⟩ ∈ [-1, 1]
```

After entangling gates, this captures **cross-feature correlations** — approximately `cos(xⱼ)·cos(xₖ)` plus interaction terms.

**Example — 4-qubit Angle circuit output for x = (0.5, 0.3, 0.8, 0.1):**
```
Feature 1 (⟨Z₀⟩):  ≈ cos(0.5) ≈  0.878
Feature 2 (⟨Z₁⟩):  ≈ cos(0.3) ≈  0.955
Feature 3 (⟨Z₂⟩):  ≈ cos(0.8) ≈  0.697
Feature 4 (⟨Z₃⟩):  ≈ cos(0.1) ≈  0.995
Feature 5 (⟨Z₀Z₁⟩): ≈ cos(0.5)·cos(0.3) + interaction ≈ 0.838
...
Feature 10 (⟨Z₂Z₃⟩): captures X₃-X₄ correlation
```

All 10 features extracted from one quantum circuit evaluation.

---

### 4.4 Circuit 1: Angle Encoding + BasicEntanglerLayers

**Step 1 — Angle Embedding:**
Apply Ry(xₖ) to each qubit k:
```
|0000⟩ → Ry(x₁)⊗Ry(x₂)⊗Ry(x₃)⊗Ry(x₄) → |ψ_encode⟩
```
State of qubit k after encoding:
```
|ψₖ⟩ = cos(xₖ/2)|0⟩ + sin(xₖ/2)|1⟩
```

**Step 2 — BasicEntanglerLayers (with zero weights):**
The zero-weight RY gates are identity (no-ops). What remains is a ring of CNOT gates:
```
CNOT(0→1), CNOT(1→2), CNOT(2→3), CNOT(3→0)
```

CNOTs create entanglement — qubit 0's state now depends on qubit 1's state. This means the measurement of `⟨Z₀Z₁⟩` will capture cross-feature information from both X₁ and X₂.

**Step 3 — Measurements:**
```
Single: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩                          (4 features)
Pairs:  ⟨Z₀Z₁⟩, ⟨Z₀Z₂⟩, ⟨Z₀Z₃⟩, ⟨Z₁Z₂⟩, ⟨Z₁Z₃⟩, ⟨Z₂Z₃⟩  (6 features)
Total: 10 quantum features per data point
```

**Functional form:** This circuit produces features in the **cosine basis**:
```
φ_angle(x) ≈ [cos(x₁), cos(x₂), cos(x₃), cos(x₄),
               cos(x₁)cos(x₂)+..., ...]
```

It captures X₁·X₃ interaction **indirectly** (through cosine products). It does **not** capture `log|X₂|`.

---

### 4.5 Circuit 2: ZZ Feature Map (Havlíček et al. 2019)

This is the most theoretically motivated quantum feature map. It implements:

**One repetition:**

Step 1 — Hadamard gates:
```
H|0⟩ = (1/√2)(|0⟩ + |1⟩)
```
Puts all qubits in equal superposition (neither |0⟩ nor |1⟩).

Step 2 — Rz(xₖ) on each qubit:
```
Rz(xₖ)(H|0⟩) = (1/√2)(e^(-ixₖ/2)|0⟩ + e^(ixₖ/2)|1⟩)
```

Step 3 — For each pair (j,k), apply a ZZ interaction:
```
CNOT(j→k) · Rz((π-xⱼ)(π-xₖ), k) · CNOT(j→k)
```

This implements the unitary `exp(i·(π-xⱼ)(π-xₖ)·Z_j⊗Z_k)`, which encodes the **product** (π-xⱼ)(π-xₖ) into the phase of the entangled state.

Expanding the product:
```
(π - xⱼ)(π - xₖ) = π² - π·xⱼ - π·xₖ + xⱼ·xₖ
```

The term `xⱼ·xₖ` is directly embedded. This means:

```
ZZ circuit contains X₁·X₃ directly in its phase encoding
```

This is why ZZ should theoretically outperform Angle for Regime 2. In practice, extracting it through Z measurements requires the right observable.

**Worked example for pair (0,1):**
```
x₁ = 0.5 (after squashing), x₂ = 0.3
ZZ angle = (π - 0.5)(π - 0.3) = 2.642 × 2.842 = 7.508
```
This angle encodes both x₁, x₂, and their interaction.

---

### 4.6 Circuit 3: IQP Encoding (Shepherd & Bremner 2009)

IQP stands for "Instantaneous Quantum Polynomial." All gates in the circuit **commute** with each other (they're all diagonal in the computational basis), meaning they can in principle all be applied simultaneously — hence "instantaneous."

**Structure:**
```
H⊗n → diagonal_phase_gates(x) → H⊗n
```

The diagonal gates apply phases:
- Single-qubit: exp(i·xₖ·Z_k)
- Two-qubit: exp(i·xⱼ·xₖ·Z_j⊗Z_k)

The second Hadamard layer converts these phases into measurable amplitudes via quantum interference.

**Why is IQP interesting?**
The output distribution of IQP circuits is believed to be classically **hard to simulate** — sampling from it is likely #P-hard (as hard as counting satisfying assignments of a Boolean formula). This suggests the feature space it accesses cannot be efficiently reproduced classically.

**In practice for our use case:**
The Fourier-like structure of IQP gives features of the form:
```
φ_IQP(x) ≈ [cos(x₁), cos(x₂), ..., cos(x₁x₂), cos(x₁x₃), ...]
```

It captures cross-feature interactions through cosine of products, similar to ZZ but with a different algebraic structure.

---

### 4.7 The Hilbert Space Argument

With n qubits, the state space has dimension 2ⁿ:
```
n=4  qubits → 16-dimensional Hilbert space
n=15 qubits → 32,768-dimensional Hilbert space
```

Classical polynomial features with 4 inputs and degree d have:
```
d=2: C(4+2,2)-1 = 14 features
d=3: C(4+3,3)-1 = 34 features
d=10: C(4+10,10)-1 = 1,001 features
```

A quantum feature map implicitly projects into a 2ⁿ-dimensional space with just n qubits. The claim is that the specific nonlinear structure of quantum mechanics accesses a fundamentally different subset of this space than polynomials do.

**However** — the features we actually *extract* are only 10 (4 + 6 expectation values), not 16. We're sampling from the Hilbert space, not reading all of it. This is an important limitation.

---

## 5. Ridge & Lasso Regression

### 5.1 Ordinary Least Squares (OLS)

Given features matrix X (n×p) and targets y (n×1), OLS minimizes:
```
L(β) = ||y - Xβ||² = Σᵢ (yᵢ - xᵢᵀβ)²
```

Closed-form solution:
```
β̂_OLS = (XᵀX)⁻¹Xᵀy
```

**Problem:** When p is large (e.g., 135 features in poly3, or 10 quantum + 14 poly2 = 24 features), `XᵀX` becomes nearly singular and the solution β̂ has enormous variance — it **overfits**.

### 5.2 Ridge Regression (L2 Regularization)

Ridge adds an L2 penalty on the magnitude of coefficients:
```
L_Ridge(β) = ||y - Xβ||² + α·||β||²
           = Σᵢ(yᵢ - xᵢᵀβ)² + α·Σⱼ βⱼ²
```

The hyperparameter α controls the tradeoff:
- α = 0: identical to OLS (can overfit)
- α → ∞: β → 0 (underfits, predicts the mean)

Closed-form solution:
```
β̂_Ridge = (XᵀX + α·I)⁻¹Xᵀy
```

The `α·I` term "inflates" the diagonal of XᵀX, making it always invertible.

**Example:** Suppose we have features (X₁, X₁², X₁·X₃, quantum_feat_1) and targets y.

Without regularization (α=0), the model might set β = (1000, -999, 1, -500) to fit training data exactly. These huge opposing coefficients cancel in-sample but generalize poorly.

With α=1.0: β might become β = (0.8, -0.3, 0.9, 0.1) — smaller, smoother, better generalization.

**Why α=1.0 for all models is wrong:**
- 4-feature raw model: 4 parameters → α=1.0 may be too strong (over-regularizes)
- 135-feature poly3 model: 135 parameters → α=1.0 may be too weak (under-regularizes)
- Correct: tune α via cross-validation separately for each model

### 5.3 Lasso Regression (L1 Regularization)

```
L_Lasso(β) = ||y - Xβ||² + α·||β||₁
           = Σᵢ(yᵢ - xᵢᵀβ)² + α·Σⱼ |βⱼ|
```

The L1 penalty drives many coefficients exactly to zero — it performs **automatic feature selection**.

Unlike Ridge (which has a closed-form solution), Lasso requires iterative optimization (coordinate descent).

**Example:** With 34 polynomial features, Lasso with α=0.01 might set all cubic terms to zero, keeping only the 14 degree-2 features that matter.

### 5.4 Why Same Model Class Is Required

The challenge mandates using Ridge for both classical and quantum features. This isolates the effect of the **feature transformation**, not the modeling choice.

Comparison matrix — fair vs unfair:

| Setup | Classical | Quantum | Is it fair? |
|---|---|---|---|
| Ridge vs Ridge | Ridge(poly2) | Ridge(poly2+Q) | ✅ Fair |
| OLS vs Ridge | LR(raw) | Ridge(Q) | ❌ Unfair — model difference |
| Ridge vs MLP | Ridge(poly) | Neural net(Q) | ❌ Unfair — model difference |

---

## 6. Evaluation Metrics

### 6.1 Mean Squared Error (MSE)

```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

Units: squared units of Y. Sensitive to outliers (large errors are penalized quadratically).

**Example:**
```
y    = [3.0, -1.0,  5.0,  2.0]
ŷ    = [2.8, -0.8,  4.5,  2.2]
errors = [0.2, -0.2, 0.5, -0.2]
MSE  = (0.04 + 0.04 + 0.25 + 0.04) / 4 = 0.0925
```

**Regime 2 specific:** Because Regime 2 has larger Y values (X₁·X₃ with mean 3×3=9 vs 2·0-0=0), even a small percentage error in Regime 2 contributes disproportionately to overall MSE. This makes per-regime breakdown essential.

### 6.2 Pearson Correlation (IC)

```
ρ(y, ŷ) = Cov(y, ŷ) / (σ_y · σ_ŷ)
         = Σᵢ(yᵢ-ȳ)(ŷᵢ-ŷ̄) / [sqrt(Σ(yᵢ-ȳ)²)·sqrt(Σ(ŷᵢ-ŷ̄)²)]
```

Range: [-1, 1]. In finance this is called the **Information Coefficient (IC)**.

Unlike MSE, correlation is **scale-invariant** — it measures ranking quality, not magnitude accuracy. For portfolio construction, ranking stocks by predicted return matters more than the exact prediction value.

**Example:**
```
y  = [0.02, -0.01, 0.03, 0.005]   (actual returns)
ŷ₁ = [0.01, -0.005, 0.02, 0.003]  (perfectly ranked)   → ρ = 1.0
ŷ₂ = [100,  -50,    200,  30  ]    (right rank, wrong scale) → ρ = 1.0
ŷ₃ = [-0.01, 0.02, 0.01, -0.003]  (reversed)           → ρ = -1.0
```

An IC of 0.05 is considered decent in finance. IC of 0.10 is very good.

### 6.3 Per-Regime Evaluation

```python
# Correct way:
mask_r1 = (r_test == 1)
mask_r2 = (r_test == 2)

mse_r1 = MSE(y_test[mask_r1], ŷ[mask_r1])
mse_r2 = MSE(y_test[mask_r2], ŷ[mask_r2])
```

If a quantum circuit captures `X₁·X₃` better than classical:
- MSE_R2(quantum) << MSE_R2(classical)
- MSE_R1(quantum) ≈ MSE_R1(classical) — linear regime, both should do equally well

This breakdown is the **most scientifically interesting** result and currently missing from the notebook.

---

## 7. Walk-Forward Backtest

### 7.1 Why Not Random Train/Test Split?

Stock returns are a **time series** — observations at time t are correlated with observations at t-1, t-2, etc. If you randomly sample train/test splits, future data leaks into the training set:

```
WRONG:
Train: [t₁, t₅, t₈, t₁₀, ...]   (contains future observations)
Test:  [t₂, t₃, t₆, t₉, ...]
```

A model trained on this can implicitly learn "what happened tomorrow" through correlated features, grossly inflating performance.

### 7.2 Rolling Window Design

```
Day 0──────────────────────────────────────────────────── Day 1259
│                                                                 │
│  Window 1:  ├─── TRAIN (504 days) ───┤ TEST (5d) │            │
│  Window 2:      ├─── TRAIN (504 days) ───┤ TEST (5d) │        │
│  Window 3:          ├─── TRAIN (504 days) ───┤ TEST (5d) │    │
│  ...                                                            │
```

At each step:
1. Fit scaler on training window only
2. Fit Ridge on training window only
3. Predict on the next 5 days
4. Collect predictions; advance window by 5 days

### 7.3 Data Leakage Sources to Avoid

| Operation | Leak? | Fix |
|---|---|---|
| `scaler.fit_transform(all_data)` | ✅ YES | Fit scaler inside loop on train only |
| `poly.fit_transform(all_data)` | ✅ YES | Fit poly inside loop on train only |
| Pre-extracting quantum features | ❌ NO | Circuit has no trainable params — ok |
| `Ridge.fit(all_data)` | ✅ YES | Fit Ridge inside loop on train only |

**The current notebook correctly pre-extracts quantum features** (no leakage since the quantum circuit has no fitted parameters). However, it fits the scaler once on the first window and reuses it — this is a minor leak that grows over time.

### 7.4 Target Construction

```
Y_{i,t} = R_{i,t→t+5} - R_{SPY,t→t+5}
```

Where:
```
R_{i,t→t+5} = (Price_{i,t+5} / Price_{i,t}) - 1   (5-day log return ≈ simple return)
```

Subtracting the market return removes systematic market exposure (beta), focusing on **alpha** — return attributable to the stock's own characteristics.

---

## 8. Why Quantum Features Are Different

### 8.1 The Kernel Perspective

The inner product between two quantum-encoded data points defines a **quantum kernel**:

```
K_Q(x, x') = |⟨ψ(x)|ψ(x')⟩|²
```

This measures how "similar" two data points look from the quantum state space perspective. The ZZ feature map produces a kernel of the form:

```
K_ZZ(x, x') = |⟨0|U†(x')U(x)|0⟩|²
```

Where U(x) is the ZZ encoding circuit. This kernel function:
- Is not efficiently computable classically (that's the claim)
- Captures features of the form exp(i·(π-xⱼ)(π-xₖ))
- Implicitly accesses 2ⁿ-dimensional Hilbert space

### 8.2 What Each Circuit Can and Cannot Capture

| Feature needed | Angle | ZZ | IQP | Poly2 | Log/Abs |
|---|---|---|---|---|---|
| X₁ (linear) | cos(X₁) | phase(X₁) | cos(X₁) | X₁ | — |
| X₁² | — | — | — | X₁² | — |
| X₁·X₃ | cos(X₁)cos(X₃) | **(π-X₁)(π-X₃) ⊃ X₁X₃** | cos(X₁X₃) | X₁X₃ | — |
| log\|X₂\| | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Yes** |
| \|X₂\| | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Yes** |

**Key takeaway:** No quantum circuit in our experiments captures `log|X₂|` — the fundamental nonlinearity in Regime 2. This is why classical poly3 still outperforms quantum on Part I. Adding log/abs to the classical baseline would raise the bar further.

### 8.3 Practical Summary of Results

| Model | MSE (Part I) | What it uses |
|---|---|---|
| LR (raw) | 4.47 | X₁, X₂, X₃, X₄ — linear only |
| Ridge (poly deg=2) | 1.63 | + X₁X₃, X₁², etc. |
| Ridge + Q Angle | 1.45 | + cos(X₁)cos(X₃) interactions |
| Ridge (poly deg=3) | **1.22** | + cubic terms, best classical bar |
| Ridge + Poly2 + Angle | 1.45 | quantum helps over poly2 alone |

The story: quantum features provide **real value over raw features** and **improvement over poly2**, but cannot surpass **poly3** without `log|X₂|` features or trained VQC weights.

---

## Summary Cheatsheet

```
Data:           X ∈ ℝ⁴, Y ∈ ℝ (regime-switching DGP)
Preprocessing:  clip → StandardScaler → clip ±5 → π·tanh(x/2) for quantum
Classical:      PolynomialFeatures(deg) → Ridge(α)
Quantum:        AngleEmbed/ZZMap/IQP → 10 expectation values → Ridge(α)
Augmented:      [classical_features, quantum_features] → Ridge(α)
Evaluation:     MSE, MAE, Pearson-ρ — per regime + aggregate
Part II:        15 stock-minus-market features → same pipeline → walk-forward backtest
```
