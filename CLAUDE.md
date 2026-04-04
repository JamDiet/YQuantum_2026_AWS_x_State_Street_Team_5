# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Challenge Context

**YQuantum 2026 — AWS x State Street: Quantum Feature Augmentation (QFA)**

Core question: Do quantum-derived feature transformations improve out-of-sample predictive performance for financial prediction tasks vs. classical feature engineering, under strict train/test separation?

Both outcomes (positive and negative) are valid. **Rigorous methodology matters more than beating the baseline.**

### Two Tasks

1. **Part I — Synthetic Regime-Switching** (10k train / 10k test): 4 features, 2-regime latent process. Y = 2X1-X2+ε (R1) or X1·X3+log(|X2|+1)+ε (R2). R1 is 75%, R2 is 25% (latent).
2. **Part II — S&P 500 Excess Returns**: 10 tickers, 15 stock-minus-market features, walk-forward backtest (504-day train, 5-day roll).

---

## Environment

```bash
# Install dependencies (run once)
%pip install -q pennylane scikit-learn scipy pandas numpy matplotlib seaborn yfinance
%pip install amazon-braket-sdk amazon-braket-pennylane-plugin
```

All work lives in Jupyter notebooks. Run via Jupyter Lab or the AWS Braket notebook environment.

**AWS Braket**: Pre-configured in Workshop Studio. Use `LocalSimulator("default")` for local state-vector simulation. Swap to `braket.aws.qubit` for QPU/managed simulators.

---

## Key Files

| File | Role |
|------|------|
| `QFA_Solution.ipynb` | **Main working notebook** — current demo solution with baselines and quantum circuits |
| `QFA_Overview.ipynb` | Reference guide — do not modify, use as documentation |
| `docs/task1_classical_baseline_plan.md` | Detailed classical baseline plan with code skeletons |
| `docs/task1_solution_evaluation.md` | Gap analysis of current solution vs. plan — **read this before iterating** |

---

## Current Solution State (Hour 2)

### What's implemented in `QFA_Solution.ipynb`

**Part I (Synthetic):**
- Data generation: `generate_regime_data(n=10000, seed=42)` — single seed only
- Preprocessing: `StandardScaler` → clip ±5σ → `π·tanh(x/2)` squash for quantum inputs
- Classical: LR (raw), Ridge poly-deg2, Ridge poly-deg3, Lasso poly-deg2 (all α hardcoded)
- Quantum circuits: Angle, ZZ, IQP — each producing 10 features (4 single-Z + 6 pairwise-ZZ)
- Augmented: Ridge(raw+Q) and Ridge(poly2+Q) for each circuit
- **Best result so far: Ridge poly-deg3 MSE=1.220, Corr=0.909**

**Part II (Stock):**
- 10 S&P 500 tickers via yfinance (`USE_REAL_DATA = True` toggle; falls back to synthetic GBM)
- 15-qubit Angle circuit → 120 quantum features
- Walk-forward: 504-day train window, 5-day roll step, `Ridge(alpha=1.0)` throughout

---

## Known Gaps to Fix (Priority Order)

### High Priority

1. **Classical baseline is incomplete** — `log(|Xi|+1)` and `|Xi|` features are missing. Regime 2 DGP uses `log(|X2|+1)` directly. Without this, classical bar is artificially low and quantum comparison is unfair. Add `log_abs_features()` and `cross_log_features()` from the baseline plan.

2. **Per-regime evaluation missing** — `r_train` and `r_test` are stored but never used. Must break down MSE/Corr separately for Regime 1 and Regime 2 subsets. This reveals whether quantum circuits preferentially capture R2's nonlinear structure.

3. **Preprocessing order is wrong for Cauchy** — Current code: `StandardScaler` → clip ±5. Correct order: clip/winsorize at percentile → then `StandardScaler`. Cauchy X2 has undefined variance; fitting StandardScaler on raw Cauchy draws contaminates mean/std estimates.

### Medium Priority

4. **Alpha is hardcoded at 1.0 everywhere** — Same α=1.0 applied to 4-feature and 135-feature models. Use `GridSearchCV` with `cv=5` on train set only. At minimum report sensitivity to α.

5. **Multi-seed stability** — All results from `seed=42` only. Add seeds [123, 7, 999, 2026] and report mean ± std. A ±0.05 swing in correlation across seeds could flip the quantum vs. classical conclusion.

6. **Preprocessing asymmetry** — Classical inputs are `X_train_s`; quantum inputs are `π·tanh(X_train_s/2)`. When building augmented matrix `np.hstack([X_train_s, Q_tr])`, apply a second `StandardScaler` over the combined matrix before Ridge to equalize regularization.

---

## Circuit Architecture Reference

| Circuit | Feature type | Directly captures X1·X3 | Directly captures log\|X2\| | Depth |
|---------|-------------|-------------------------|------------------------------|-------|
| Angle | Trigonometric (cosine basis) | Indirectly | No | 2 |
| ZZ | Polynomial-phase products — `(π-xi)(π-xj) ⊃ xi·xj` | **Yes** | No | 31 |
| IQP | Fourier/Boolean coefficients | Indirectly | No | 1 |
| Poly-deg2 (classical) | Polynomial xi·xj | **Yes** | No | — |
| Log/abs (missing from baseline) | Monotone nonlinear | No | **Yes** | — |

**Key implication**: No quantum circuit captures `log|X2|`. Adding log/abs to classical raises the bar before claiming quantum advantage.

---

## Experimental Design Rules (from challenge spec)

- **Same model class** for classical and quantum (Ridge throughout — enforced)
- **Same regularization strategy** for classical and quantum (currently violated — α not tuned)
- **No look-ahead**: In Part II, fit StandardScaler and poly features on train window only; never `fit_transform` on test
- Report both positive and negative results with equal rigor
- Report quantum resource usage: qubit count, circuit depth, gate count

---

## Iteration Directions

After fixing gaps above, push forward with:

- **VQC with trained weights**: Replace `weights=np.zeros((2,4))` with learned params (parameter-shift gradients)
- **Probability extraction**: Extract all `2^4=16` basis probabilities instead of 10 expectation values (richer but monitor overfitting)
- **Pauli X/Y observables**: Add `⟨Xi⟩`, `⟨Yi⟩` — 8 more features per circuit with different functional dependencies
- **More qubits than features**: Map 4 inputs → 8 qubits with repeated angle encoding (larger Hilbert space)
- **Feature correlation check**: Compute pairwise correlation between Angle, ZZ, IQP feature matrices to check redundancy

---

## Evaluation Metrics (must report all)

- MSE, MAE, Pearson correlation (aggregate)
- Per-regime breakdown (R1 and R2 separately)
- Train vs. test MSE gap (overfitting diagnostic)
- Mean ± std across 5 seeds
- Part II: Information Coefficient (IC), IC stability over time
- Quantum resource report: qubit count, depth, gate count per circuit
