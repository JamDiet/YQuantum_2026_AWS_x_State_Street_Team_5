#!/usr/bin/env python
# coding: utf-8

# <p align="center">
#   <img src="https://www.qusecure.com/wp-content/uploads/2023/07/Amazon-Web-Services-AWS-Emblem.png" width="170" height="80"/>
# </p>
# 
# # YQuantum 2026 — AWS × State Street Challenge
# ## Quantum Feature Augmentation for Financial Market Prediction
# 
# **Core Question:** Do quantum-derived feature transformations improve out-of-sample predictive performance for financial prediction tasks, relative to classical feature engineering, under strict train/test separation and rolling backtests?
# 
# | Section | Content |
# |---|---|
# | 1. Setup | Imports, device configuration |
# | 2. Part I – Synthetic | Regime-switching DGP, classical & quantum baselines |
# | 3. Part II – Real Stock Data | S&P 500 excess returns, walk-forward backtest |
# | 4. Quantum Resources | Circuit depth, qubit count, cost–performance tradeoff |
# | 5. Discussion | Conclusions, honest appraisal, future work |
# 

# ---
# ## 1. Environment Setup

# In[ ]:


# Uncomment to install on fresh environment
# %pip install -q pennylane scikit-learn scipy pandas numpy matplotlib seaborn yfinance
# %pip install amazon-braket-sdk amazon-braket-pennylane-plugin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

import pennylane as qml

np.random.seed(42)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

# ─── DEVICE CONFIGURATION ──────────────────────────────────────────────────
USE_BRAKET = False  # Set True when running on AWS Workshop Studio

if USE_BRAKET:
    device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
    
    dev4  = qml.device("braket.aws.qubit", device_arn=device_arn, wires=4)
    dev15 = qml.device("braket.aws.qubit", device_arn=device_arn, wires=15)
else:
    dev4  = qml.device("default.qubit", wires=4)
    dev15 = qml.device("default.qubit", wires=15)

print(f"Device: {dev4}")
print(f"PennyLane version: {qml.__version__}")


# ---
# ## 2. Part I — Simulated Regime Switching Process

# ### 2.1 Data Generating Process
# 
# The target $Y$ comes from a **latent two-regime model**:
# 
# $$Y = \begin{cases} 2X_1 - X_2 + \varepsilon & \text{Regime 1 (75\%)} \\ X_1 \cdot X_3 + \log(|X_2|+1) + \varepsilon & \text{Regime 2 (25\%)} \end{cases}$$
# 
# Regime 2 features correlated $X_1, X_3$, heavy-tailed Cauchy $X_2$, and exponential $X_4$ (pure noise). The regime indicator is **latent** — the model must discover it from features alone.
# 

# In[ ]:


def generate_regime_data(n=10000, seed=42):
    rng = np.random.RandomState(seed)
    regime = rng.choice([1, 2], size=n, p=[0.75, 0.25])
    X = np.zeros((n, 4))
    Y = np.zeros(n)
    for i in range(n):
        if regime[i] == 1:
            X[i, 0] = rng.normal(0, 1)          # X1 — linear signal R1
            X[i, 1] = rng.normal(0, 1)          # X2 — linear R1, nonlinear R2
            X[i, 2] = rng.normal(0, 1)          # X3 — only relevant in R2
            X[i, 3] = rng.uniform(-1, 1)        # X4 — pure noise
            Y[i] = 2*X[i,0] - X[i,1] + rng.normal(0,1)
        else:
            x1x3 = rng.multivariate_normal([3,3], [[1,0.8],[0.8,1]])
            X[i, 0] = x1x3[0]
            X[i, 2] = x1x3[1]
            X[i, 1] = np.clip(rng.standard_cauchy(), -10, 10)  # heavy-tailed
            X[i, 3] = rng.exponential(1)
            Y[i] = X[i,0]*X[i,2] + np.log(np.abs(X[i,1])+1) + rng.normal(0,1)
    return X, Y, regime

X_train, y_train, r_train = generate_regime_data(10000, seed=42)
X_test,  y_test,  r_test  = generate_regime_data(10000, seed=123)

print(f"Train: {X_train.shape}  |  Regime 1: {(r_train==1).mean():.1%}  Regime 2: {(r_train==2).mean():.1%}")
print(f"Test:  {X_test.shape}   |  y_train mean={y_train.mean():.2f}  std={y_train.std():.2f}")

# Visualise data
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(X_train[r_train==1, 0], y_train[r_train==1], alpha=0.1, s=3, c='steelblue', label='Regime 1')
axes[0].scatter(X_train[r_train==2, 0], y_train[r_train==2], alpha=0.3, s=3, c='tomato', label='Regime 2')
axes[0].set_xlabel("X1"); axes[0].set_ylabel("Y"); axes[0].set_title("Y vs X1 by Regime"); axes[0].legend()
axes[1].scatter(X_train[r_train==1, 0]*X_train[r_train==1, 2], y_train[r_train==1], alpha=0.1, s=3, c='steelblue')
axes[1].scatter(X_train[r_train==2, 0]*X_train[r_train==2, 2], y_train[r_train==2], alpha=0.3, s=3, c='tomato')
axes[1].set_xlabel("X1 × X3"); axes[1].set_ylabel("Y"); axes[1].set_title("Y vs X1·X3 (interaction, key in R2)")
axes[2].hist(y_train[r_train==1], bins=60, alpha=0.6, density=True, color='steelblue', label='R1')
axes[2].hist(y_train[r_train==2], bins=60, alpha=0.6, density=True, color='tomato', label='R2')
axes[2].set_xlabel("Y"); axes[2].set_title("Y distribution by Regime"); axes[2].legend()
plt.tight_layout(); plt.savefig("regime_data.png", bbox_inches='tight'); plt.show()


# ### 2.2 Preprocessing

# In[ ]:


# ── FIXED: clip BEFORE StandardScaler (Cauchy X2 has undefined variance) ──
# Wrong order was: StandardScaler → clip  (Cauchy extremes contaminate mean/std)
# Correct order:   clip → StandardScaler → clip again for safety

CLIP_PERCENTILE = 1  # winsorise at 1st/99th percentile before scaling

def preprocess(X_tr, X_te):
    """Correct preprocessing: percentile clip → StandardScaler → safety clip → tanh squash."""
    # Step 1: winsorise BEFORE fitting scaler (prevents Cauchy from poisoning mean/std)
    lo = np.percentile(X_tr, CLIP_PERCENTILE, axis=0)
    hi = np.percentile(X_tr, 100 - CLIP_PERCENTILE, axis=0)
    X_tr_clipped = np.clip(X_tr, lo, hi)
    X_te_clipped = np.clip(X_te, lo, hi)  # use TRAIN percentiles on test

    # Step 2: StandardScaler fitted on clipped train data only
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_clipped)
    X_te_s = sc.transform(X_te_clipped)

    # Step 3: safety clip at ±5σ
    X_tr_s = np.clip(X_tr_s, -5, 5)
    X_te_s = np.clip(X_te_s, -5, 5)

    return X_tr_s, X_te_s

def restrict_range(data, median: float, mad: float):
    return np.pi * np.tanh((data - median) / mad)

X_train_s, X_test_s = preprocess(X_train, X_test)

median = np.median(X_train_s)
mad = np.median(abs(X_train_s - median))

# Restrict data range to [-pi, pi]
X_train_s = restrict_range(X_train_s, median, mad)
X_train_q = restrict_range(X_train_s, median, mad)
X_test_s = restrict_range(X_test_s, median, mad)
X_test_q = restrict_range(X_test_s, median, mad)

print(f"Scaled range:  [{X_train_s.min():.2f}, {X_train_s.max():.2f}]")
print(f"Quantum range: [{X_train_q.min():.2f}, {X_train_q.max():.2f}]  (bounded for RY gates)")

# Check Cauchy column (X2) before vs after
x2_raw_std = X_train[:, 1].std()
x2_fix_std = X_train_s[:, 1].std()
print(f"X2 std (raw): {x2_raw_std:.2f}  →  X2 std (fixed): {x2_fix_std:.2f}  (Cauchy tamed)")


# ### 2.3 Classical Baselines

# In[ ]:


def evaluate(y_true, y_pred, label):
    """Return MSE, MAE, Pearson correlation and model label as a dict."""
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else float('nan')
    return {"Model": label, "MSE": round(mse, 4), "MAE": round(mae, 4), "Corr": round(corr, 4)}

def tune_ridge(X_tr, y_tr):
    """Tune Ridge alpha via 5-fold CV using a logarithmic search space."""
    # Searching powers of 10 ensures we find the right magnitude of regularization
    alphas = np.logspace(-3, 3, 13) 
    gs = GridSearchCV(Ridge(), {"alpha": alphas}, cv=5,
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_["alpha"]

def tune_lasso(X_tr, y_tr):
    """Tune Lasso alpha via 5-fold CV."""
    # Lasso is often more sensitive; starting at a smaller alpha
    alphas = np.logspace(-4, 1, 13)
    gs = GridSearchCV(Lasso(max_iter=10000), {"alpha": alphas}, cv=5,
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_["alpha"]

# ── 1. LOG/ABS FEATURES ──
def add_log_abs_features(X):
    """Add log(|Xi|+1) and |Xi| — captures Regime 2 nonlinearity classically."""
    log_abs = np.log(np.abs(X) + 1)
    abs_X   = np.abs(X)
    return np.hstack([X, log_abs, abs_X])

Xtr_log = add_log_abs_features(X_train_s)
Xte_log = add_log_abs_features(X_test_s)

results_p1 = []

# 1. Raw LR
lr = LinearRegression().fit(X_train_s, y_train)
results_p1.append(evaluate(y_test, lr.predict(X_test_s), "LR (raw)"))

# 2. Log+Abs features → tuned Ridge
ridge_log, alpha_log = tune_ridge(Xtr_log, y_train)
results_p1.append(evaluate(y_test, ridge_log.predict(Xte_log), f"Ridge (log+abs, α={alpha_log:.3f})"))
print(f"Ridge log+abs best alpha: {alpha_log:.3f}")

# 3. Poly deg=2 → tuned Ridge
poly2 = PolynomialFeatures(degree=2, include_bias=False)
Xtr_p2 = poly2.fit_transform(X_train_s);  Xte_p2 = poly2.transform(X_test_s)
ridge_p2, alpha_p2 = tune_ridge(Xtr_p2, y_train)
results_p1.append(evaluate(y_test, ridge_p2.predict(Xte_p2), f"Ridge (poly deg=2, α={alpha_p2:.3f})"))
print(f"Ridge poly2 best alpha: {alpha_p2:.3f}")

# 4. Poly deg=3 → tuned Ridge
poly3 = PolynomialFeatures(degree=3, include_bias=False)
Xtr_p3 = poly3.fit_transform(X_train_s);  Xte_p3 = poly3.transform(X_test_s)
ridge_p3, alpha_p3 = tune_ridge(Xtr_p3, y_train)
results_p1.append(evaluate(y_test, ridge_p3.predict(Xte_p3), f"Ridge (poly deg=3, α={alpha_p3:.3f})"))
print(f"Ridge poly3 best alpha: {alpha_p3:.3f}")

# 5. Poly2 + Log+Abs → tuned Ridge
Xtr_p2log = np.hstack([Xtr_p2, add_log_abs_features(X_train_s)[:, 4:]])
Xte_p2log = np.hstack([Xte_p2, add_log_abs_features(X_test_s)[:, 4:]])
ridge_p2log, alpha_p2log = tune_ridge(Xtr_p2log, y_train)
results_p1.append(evaluate(y_test, ridge_p2log.predict(Xte_p2log),
                            f"Ridge (poly2+log+abs, α={alpha_p2log:.3f})"))
print(f"Ridge poly2+log+abs best alpha: {alpha_p2log:.3f}")

# 6. Lasso deg=2 (Now tuned!)
lasso_p2, alpha_lasso = tune_lasso(Xtr_p2, y_train)
results_p1.append(evaluate(y_test, lasso_p2.predict(Xte_p2), f"Lasso (poly deg=2, α={alpha_lasso:.3f})"))
print(f"Lasso poly2 best alpha: {alpha_lasso:.3f}")

df_cl = pd.DataFrame(results_p1)
print("\n=== Classical Baselines (Out-of-Sample) ===")
print(df_cl.sort_values("MSE"))


# In[ ]:


# ==============================================================================
# STRICT TANH CONTROLS (Run this right after your classical baselines)
# ==============================================================================

# 1. Raw LR (Tanh Control)
lr_q = LinearRegression().fit(X_train_q, y_train)
results_p1.append(evaluate(y_test, lr_q.predict(X_test_q), "LR (raw Tanh Control)"))

# 2. Poly deg=2 → tuned Ridge (Tanh Control)
# This explicitly uses the `poly2` and `tune_ridge` functions you already defined!
Xtr_p2_q = poly2.fit_transform(X_train_q)
Xte_p2_q = poly2.transform(X_test_q)
ridge_p2_q, alpha_p2_q = tune_ridge(Xtr_p2_q, y_train)
results_p1.append(evaluate(y_test, ridge_p2_q.predict(Xte_p2_q), f"Ridge (poly deg=2 Tanh Control, α={alpha_p2_q})"))

# Build and display the final combined DataFrame
df_cl = pd.DataFrame(results_p1)
print("\n=== Final Classical Baselines (With Controls) ===")
print(df_cl.sort_values("MSE"))


# ### 2.4 Quantum Feature Maps

# We implement three distinct quantum encoding strategies, each producing **10 features** per data point (4 single-qubit ⟨Z⟩ + 6 two-qubit ⟨ZZ⟩ expectation values):
# 
# | Circuit | Encoding | Key idea |
# |---|---|---|
# | **Angle + Entangle** | `AngleEmbedding(Y)` → `BasicEntanglerLayers` | Rotations + CNOT ring; classical speed, captures cross-qubit correlations |
# | **ZZ Feature Map** | H → Rz(xᵢ) → ZZ(xᵢxⱼ) (×2 reps) | Havlíček et al. 2019; cross-feature products in exponent |
# | **IQP Encoding** | Diagonal commuting gates (×2 reps) | Classically hard to simulate; different algebraic structure |
# 

# In[3]:


N_QUBITS = 4

# ── Circuit: Angle Encoding + ZZ Entanglement ───────────────────────────
@qml.qnode(dev4, interface="numpy")
def angle_circuit(x):
    # Step 1: Angle embedding — Ry then Rz per qubit, no entanglement
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)
    # Step 2: ZZ feature map in series (entanglement comes from here)
    for rep in range(2):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(N_QUBITS):
            for j in range(i+1, N_QUBITS):
                qml.CNOT(wires=[i, j])
                qml.RZ((np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])
    single = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    pairs  = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
              for i in range(N_QUBITS) for j in range(i+1, N_QUBITS)]
    return single + pairs

# ── Quantum Resource Metrics ─────────────────────────────────────────────
specs = qml.specs(angle_circuit)(X_train_q[0])
r = specs["resources"]
print(f"Angle+ZZ  qubits={r.num_wires}  depth={r.depth}  gates={r.num_gates}  output_features=10")
print(qml.draw(angle_circuit)(X_train_q[0]))


# In[ ]:


import time

def extract_q_features(circuit_fn, X, label=""):
    t0 = time.time()
    feats = np.array([circuit_fn(x) for x in X])
    elapsed = time.time() - t0
    print(f"  {label}: {len(X):,} samples → {elapsed:.1f}s ({elapsed/len(X)*1000:.1f}ms/sample)")
    return feats

print("Extracting quantum features (full 10k train + 10k test)...")
Q_tr_angle = extract_q_features(angle_circuit, X_train_q, "Angle train")
Q_te_angle = extract_q_features(angle_circuit, X_test_q,  "Angle test ")
print(f"\nFeature shapes: {Q_tr_angle.shape}  (N_samples × 10 quantum features)")


# ### 2.5 Results — Synthetic Task

# In[ ]:


from sklearn.model_selection import GridSearchCV

# ── Quantum-only augmentation (with re-standardized combined matrix) ──────
ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]

def augment_and_fit(X_tr_cl, X_te_cl, Q_tr, Q_te, y_tr, label):
    """
    Stack classical + quantum features, apply a second StandardScaler
    over the combined matrix, then tune Ridge alpha via CV.
    This ensures equal regularization across feature types.
    """
    Xaug_tr = np.hstack([X_tr_cl, Q_tr])
    Xaug_te = np.hstack([X_te_cl, Q_te])
    # FIX: re-standardize the combined matrix so Ridge treats all features equally
    sc2 = StandardScaler()
    Xaug_tr = sc2.fit_transform(Xaug_tr)
    Xaug_te = sc2.transform(Xaug_te)
    ridge, alpha = tune_ridge(Xaug_tr, y_tr)
    return ridge.predict(Xaug_te), alpha

for name, Q_tr, Q_te in [("Angle+ZZ", Q_tr_angle, Q_te_angle)]:
    preds, alpha = augment_and_fit(X_train_s, X_test_s, Q_tr, Q_te, y_train, name)
    results_p1.append(evaluate(y_test, preds, f"Ridge+{name} (α={alpha})"))

# Poly2 + Quantum
for name, Q_tr, Q_te in [("Angle+ZZ", Q_tr_angle, Q_te_angle)]:
    preds, alpha = augment_and_fit(Xtr_p2, Xte_p2, Q_tr, Q_te, y_train, name)
    results_p1.append(evaluate(y_test, preds, f"Ridge+Poly2+{name} (α={alpha})"))

df_all = pd.DataFrame(results_p1)
print(df_all.sort_values("MSE"))

# ── FIX: Per-Regime Evaluation ─────────────────────────────────────────────
print("\n=== Per-Regime Breakdown ===")
mask1 = (r_test == 1);  mask2 = (r_test == 2)
best_models = df_all.sort_values("MSE").head(6)["Model"].tolist()

regime_rows = []
all_preds = {}
# Re-generate predictions for top models
for name, Q_tr, Q_te in [("Angle+ZZ", Q_tr_angle, Q_te_angle)]:
    Xaug_tr = StandardScaler().fit_transform(np.hstack([Xtr_p2, Q_tr]))
    Xaug_te = StandardScaler().fit(np.hstack([Xtr_p2, Q_tr])).transform(np.hstack([Xte_p2, Q_te]))
    sc2 = StandardScaler()
    Xaug_tr = sc2.fit_transform(np.hstack([Xtr_p2, Q_tr]))
    Xaug_te = sc2.transform(np.hstack([Xte_p2, Q_te]))
    ridge_q, _ = tune_ridge(Xaug_tr, y_train)
    all_preds[f"Q+Poly2+{name}"] = ridge_q.predict(Xaug_te)

all_preds["Classical Ridge poly3"] = ridge_p3.predict(Xte_p3)
all_preds["Classical poly2+log+abs"] = ridge_p2log.predict(Xte_p2log)

for mname, preds in all_preds.items():
    r1 = evaluate(y_test[mask1], preds[mask1], f"{mname} R1")
    r2 = evaluate(y_test[mask2], preds[mask2], f"{mname} R2")
    regime_rows.append({"Model": mname,
                        "R1_MSE": r1["MSE"], "R1_Corr": r1["Corr"],
                        "R2_MSE": r2["MSE"], "R2_Corr": r2["Corr"],
                        "R2_vs_R1_MSE": round(r2["MSE"]/r1["MSE"], 2)})

df_regime = pd.DataFrame(regime_rows)
print(f"  Regime 1 test samples: {mask1.sum()}  |  Regime 2 test samples: {mask2.sum()}")
print(df_regime)

# ── FIX: Multi-Seed Stability ──────────────────────────────────────────────
print("\n=== Multi-Seed Stability (5 seeds) ===")
SEEDS = [42, 123, 7, 999, 2026]
seed_results = {name: [] for name in ["LR (raw)", "Ridge poly2", "Ridge poly3",
                                       "Ridge log+abs", "Ridge+Poly2+Angle"]}

for seed in SEEDS:
    Xtr_s, ytr_s, rtr_s = generate_regime_data(10000, seed=seed)
    Xte_s, yte_s, rte_s = generate_regime_data(10000, seed=seed+1000)
    Xtr_sc, Xte_sc = preprocess(Xtr_s, Xte_s)
    Xtr_qc, Xte_qc = Xtr_sc, Xte_sc

    # LR raw
    lr_s = LinearRegression().fit(Xtr_sc, ytr_s)
    seed_results["LR (raw)"].append(pearsonr(yte_s, lr_s.predict(Xte_sc))[0])

    # Ridge poly2
    p2s = PolynomialFeatures(degree=2, include_bias=False)
    Xtrp2s = p2s.fit_transform(Xtr_sc);  Xtep2s = p2s.transform(Xte_sc)
    r2s, _ = tune_ridge(Xtrp2s, ytr_s)
    seed_results["Ridge poly2"].append(pearsonr(yte_s, r2s.predict(Xtep2s))[0])

    # Ridge poly3
    p3s = PolynomialFeatures(degree=3, include_bias=False)
    Xtrp3s = p3s.fit_transform(Xtr_sc);  Xtep3s = p3s.transform(Xte_sc)
    r3s, _ = tune_ridge(Xtrp3s, ytr_s)
    seed_results["Ridge poly3"].append(pearsonr(yte_s, r3s.predict(Xtep3s))[0])

    # Ridge log+abs
    Xtrls = add_log_abs_features(Xtr_sc);  Xtels = add_log_abs_features(Xte_sc)
    rls, _ = tune_ridge(Xtrls, ytr_s)
    seed_results["Ridge log+abs"].append(pearsonr(yte_s, rls.predict(Xtels))[0])

    # Quantum Angle+Poly2
    Qtr_s = extract_q_features(angle_circuit, Xtr_qc)
    Qte_s = extract_q_features(angle_circuit, Xte_qc)
    sc2s = StandardScaler()
    Xaug_trs = sc2s.fit_transform(np.hstack([Xtrp2s, Qtr_s]))
    Xaug_tes = sc2s.transform(np.hstack([Xtep2s, Qte_s]))
    rqs, _ = tune_ridge(Xaug_trs, ytr_s)
    seed_results["Ridge+Poly2+Angle"].append(pearsonr(yte_s, rqs.predict(Xaug_tes))[0])

seed_df = pd.DataFrame([
    {"Model": k, "Mean_Corr": round(np.mean(v),4), "Std_Corr": round(np.std(v),4),
     "Min": round(min(v),4), "Max": round(max(v),4)}
    for k, v in seed_results.items()
])
print(seed_df)

# ── Result Chart ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
df_plot = df_all.sort_values("MSE")

color_map = []
for m in df_plot["Model"]:
    if "Poly2+" in m or "poly2+" in m or "log+abs" in m: color_map.append('#6ABF69')
    elif "Ridge+" in m or "Q" in m: color_map.append('#E24A33')
    else: color_map.append('#4878CF')

axes[0].barh(df_plot["Model"], df_plot["MSE"], color=color_map)
axes[0].set_xlabel("MSE (lower is better)"); axes[0].set_title("Part I — Out-of-Sample MSE")
axes[0].axvline(df_plot["MSE"].min(), ls='--', c='gray', alpha=0.5)

axes[1].barh(df_plot["Model"], df_plot["Corr"], color=color_map)
axes[1].set_xlabel("Pearson Correlation (higher is better)")
axes[1].set_title("Part I — Out-of-Sample Correlation")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4878CF', label='Classical (raw/poly)'),
                   Patch(facecolor='#E24A33', label='Quantum only'),
                   Patch(facecolor='#6ABF69', label='Quantum + Classical')]
axes[0].legend(handles=legend_elements, loc='lower right')
plt.tight_layout(); plt.savefig("part1_results.png", bbox_inches='tight'); plt.show()




# ---
# ## 3. Part II — Predicting S&P 500 Excess Returns

# ### 3.1 Data Acquisition
# 
# **Target**: Next 5-day excess return of stock $i$:
# $$Y_{i,t} = R_{i,t}^{(5d)} - R_{\text{SP500},t}^{(5d)}$$
# 
# This removes market-wide effects and focuses on **relative stock performance**.
# 

# In[ ]:


# ── 3.1 Configuration & Data Acquisition ───────────────────────────────
import pandas as pd
import numpy as np

USE_REAL_DATA = True  # Set True on AWS

MARKET = "SPY"
PROXIES = ["SMH", "XLK", "XLY", "XLF"]

# Hardcoded configurations as requested
BUCKET_A_WEIGHTS = {
  "NVDA": 7.58,
  "AVGO": 2.64,
  "MSFT": 4.87,
  "AAPL": 6.66,
  "AMZN": 3.65,
  "GOOGL": 5.52,   # Alphabet total = GOOGL + GOOG
  "META": 2.25,
}
BUCKET_B_WEIGHTS = {
  "TSLA": 1.90,
  "BRK-B": 1.56,
  "JPM": 1.39,
}
ALL_TICKERS = [MARKET] + PROXIES + list(BUCKET_A_WEIGHTS.keys()) + list(BUCKET_B_WEIGHTS.keys())
START, END = "2020-01-01", "2025-12-31"

if USE_REAL_DATA:
    try:
        import yfinance as yf
        raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)
        # We keep the MultiIndex format: raw["Close"]["NVDA"]
        data_source = "yfinance (real)"
        print(f"Downloaded {len(raw)} rows for {len(ALL_TICKERS)} tickers from yfinance")
    except Exception as e:
        print(f"yfinance failed ({e}) — falling back to synthetic data")
        USE_REAL_DATA = False

if not USE_REAL_DATA:
    print("Generating synthetic MultiIndex market data...")
    N_DAYS = 1259
    rng2 = np.random.RandomState(42)
    dates = pd.bdate_range("2020-01-02", periods=N_DAYS)
    
    # Create a synthetic MultiIndex DataFrame to mimic yfinance
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, ALL_TICKERS], names=["Price", "Ticker"])
    raw = pd.DataFrame(index=dates, columns=cols)
    
    for t in ALL_TICKERS:
        # Simple random walk for fallback
        rets = rng2.normal(0.0005, 0.02, N_DAYS)
        close = 100 * np.exp(np.cumsum(rets))
        raw["Close", t] = close
        raw["Open", t] = close * (1 + rng2.normal(0, 0.005, N_DAYS))
        raw["High", t] = raw[["Open", "Close"]].xs(t, level=1, axis=1).max(axis=1) * 1.01
        raw["Low", t] = raw[["Open", "Close"]].xs(t, level=1, axis=1).min(axis=1) * 0.99
        raw["Volume", t] = 5e7 * (1 + np.abs(rng2.normal(0, 0.5, N_DAYS)))
    data_source = "synthetic fallback"

print(f"Data source: {data_source}")



# ── 3.1.5 Helper Functions & Bucket Construction ───────────────────────

# Normalize weights so they sum to 1.0
w_A = {k: v / sum(BUCKET_A_WEIGHTS.values()) for k, v in BUCKET_A_WEIGHTS.items()}
w_B = {k: v / sum(BUCKET_B_WEIGHTS.values()) for k, v in BUCKET_B_WEIGHTS.items()}

def compute_rsi(close_series, period=10):
    delta = close_series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_bucket(df_raw, weights):
    """Reconstructs a synthetic bucket index strictly from constituent returns to avoid price-distortion."""
    idx = df_raw.index
    bucket_ret = pd.Series(0.0, index=idx)
    bucket_dv = pd.Series(0.0, index=idx)
    open_gap  = pd.Series(0.0, index=idx)
    
    for tic, w in weights.items():
        # Sum weighted returns
        ret_i = df_raw["Close"][tic].pct_change()
        bucket_ret += w * ret_i.fillna(0)
        # Sum weighted dollar volume
        bucket_dv += w * (df_raw["Close"][tic] * df_raw["Volume"][tic])
        # Weighted open-gap (open vs prev close)
        g = (df_raw["Open"][tic] / df_raw["Close"][tic].shift(1)) - 1
        open_gap += w * g.fillna(0)
        
    # Reconstruct Close price starting at 100
    bucket_close = [100.0]
    for r in bucket_ret.iloc[1:]:
        bucket_close.append(bucket_close[-1] * (1 + r))
    
    # Reconstruct Open price: prev_close * (1 + open_gap)
    bucket_open = [bucket_close[0]]
    for i in range(1, len(bucket_close)):
        bucket_open.append(bucket_close[i-1] * (1 + open_gap.iloc[i]))
        
    bucket_df = pd.DataFrame(index=idx)
    bucket_df["Close"] = bucket_close
    bucket_df["Open"]  = bucket_open
    bucket_df["Dollar_Volume"] = bucket_dv
    return bucket_df

print("Building Synthetic Buckets...")
bA_df = build_bucket(raw, w_A)
bB_df = build_bucket(raw, w_B)
print("Buckets constructed successfully.")


# ### 3.2 Feature Engineering (15 stock-minus-market features)

# In[ ]:


# ── 3.2 Feature Engineering (Exactly 16 Features per Bucket) ───────────

def extract_16_features(bucket_df, proxies_df, name, proxy1, proxy2):
    f = pd.DataFrame(index=bucket_df.index)
    bc = bucket_df["Close"]
    bo = bucket_df["Open"]
    bdv = bucket_df["Dollar_Volume"]
    spy_c = proxies_df["Close"]["SPY"]
    spy_o = proxies_df["Open"]["SPY"]
    spy_dv = proxies_df["Close"]["SPY"] * proxies_df["Volume"]["SPY"]
    p1_c = proxies_df["Close"][proxy1]
    p2_c = proxies_df["Close"][proxy2]
    
    # helper for n-day return
    def ret(series, n): return series.pct_change(n)
    
    # Q1: Relative Momentum
    f[f"ret_5_{name}_minus_SPY"] = ret(bc, 5) - ret(spy_c, 5)
    f[f"ret_60_{name}_minus_SPY"] = ret(bc, 60) - ret(spy_c, 60)
    f[f"ret_120_{name}_minus_SPY"] = ret(bc, 120) - ret(spy_c, 120)

    # Illiquidity indicator
    abs_ret_tic = bc.pct_change().abs()
    abs_ret_spy = spy_c.pct_change().abs()
    illiq_tic = (abs_ret_tic / (bc + 1e-9)).rolling(20).mean()
    illiq_spy = (abs_ret_spy / (spy_dv + 1e-9)).rolling(20).mean()
    f[f"illiq_20_{name}_over_SPY"] = illiq_tic / illiq_spy

    # Intra-day gap
    gap_b = (bo - bc.shift(1)) / bo
    gap_s = (spy_o - spy_c.shift(1)) / spy_o
    b_z = (gap_b - gap_b.rolling(20).mean()) / (gap_b.rolling(20).std() + 1e-9)
    gap_z = (gap_s - gap_s.rolling(20).mean()) / (gap_s.rolling(20).std() + 1e-9)
    f[f"gap_zscore_diff_{name}_vs_SPY"] = b_z - gap_z
    
    # Q2: Internal State
    f[f"rsi10_{name}_minus_SPY"] = compute_rsi(bc) - compute_rsi(spy_c)
    
    b_trend = (bc.rolling(10).mean() / bc.rolling(50).mean()) - 1
    s_trend = (spy_c.rolling(10).mean() / spy_c.rolling(50).mean()) - 1
    f[f"trend10_50_{name}_minus_SPY"] = b_trend - s_trend
    
    b_hipos = (bc.rolling(20).max() / bc) - 1
    s_hipos = (spy_c.rolling(20).max() / spy_c) - 1
    f[f"highpos20_{name}_minus_SPY"] = b_hipos - s_hipos
    
    # Dollar Volume Z-Score (60 day lookback)
    dv_ma5 = bdv.rolling(5).mean()
    b_dvz = (dv_ma5 - dv_ma5.rolling(60).mean()) / (dv_ma5.rolling(60).std() + 1e-9)
    spy_dv = spy_c * proxies_df["Volume"]["SPY"]
    s_dv_ma5 = spy_dv.rolling(5).mean()
    s_dvz = (s_dv_ma5 - s_dv_ma5.rolling(60).mean()) / (s_dv_ma5.rolling(60).std() + 1e-9)
    f[f"dv_zscore_5_{name}_minus_SPY"] = b_dvz - s_dvz
    
    # Q3: External Regime Proxies
    f[f"ret_20_{proxy1}_minus_SPY"] = ret(p1_c, 20) - ret(spy_c, 20)
    f[f"ret_60_{proxy1}_minus_SPY"] = ret(p1_c, 60) - ret(spy_c, 60)
    f[f"ret_20_{proxy2}_minus_SPY"] = ret(p2_c, 20) - ret(spy_c, 20)
    f[f"ret_60_{proxy2}_minus_SPY"] = ret(p2_c, 60) - ret(spy_c, 60)
    
    # Q4: Orthogonal Enrichment
    # 120d momentum — extends Q1's lookback dimension; directly listed in README
    f[f"ret_120_{name}_minus_SPY"] = ret(bc, 120) - ret(spy_c, 120)

    # 20d realized volatility ratio — orthogonal to direction; vol regime signal
    b_vol20 = bc.pct_change().rolling(20).std()
    s_vol20 = spy_c.pct_change().rolling(20).std()
    f[f"vol20_{name}_minus_SPY"] = b_vol20 - s_vol20

    # 5-day volume-weighted return — price x vol interaction; README-listed
    spy_dv = spy_c * proxies_df["Volume"]["SPY"]
    def vw_return(close_series, vol_series, n=5):
        daily_ret = close_series.pct_change()
        return (daily_ret * vol_series).rolling(n).sum() / (vol_series.rolling(n).sum() + 1e-9)
    f[f"vwret_5_{name}_minus_SPY"] = vw_return(bc, bdv) - vw_return(spy_c, spy_dv)

    # Dollar-volume trend MA10/MA50 — directional vol trend vs. the spike Z-score in Q2
    b_dvtrend = (bdv.rolling(10).mean() / (bdv.rolling(50).mean() + 1e-9)) - 1
    s_dvtrend = (spy_dv.rolling(10).mean() / (spy_dv.rolling(50).mean() + 1e-9)) - 1
    f[f"dvtrend10_50_{name}_minus_SPY"] = b_dvtrend - s_dvtrend
    
    # Target: forward 5d excess return
    f["target"] = (bc.shift(-5) / bc - 1) - (spy_c.shift(-5) / spy_c - 1)
    
    return f.dropna() # Crucial: drops the NaN rows caused by 100-day rolling windows!

# Build final DataFrames
print("Extracting 16 Features per Bucket...")
feat_A = extract_16_features(bA_df, raw, "BA", "SMH", "XLK")
feat_B = extract_16_features(bB_df, raw, "BB", "XLY", "XLF")

print(f"Bucket A ready: {feat_A.shape}")
print(f"Bucket B ready: {feat_B.shape}")


# ### 3.2.5 Feature Quality Diagnostics
# 
# Runs **before** the walk-forward backtest and quantum circuits. Produces:
# - Pairwise correlation heatmap (gold border = |r| > 0.80)
# - Rolling 60d IC per feature (dashed = flagged)
# - Diagnostic table: Std Dev, Skew, IC, IC t-stat, Max Pair Corr, VIF, Flag
# 
# Flags: `HIGH_CORR` |r|>0.80 · `HIGH_VIF` VIF>10 · `NO_IC` |IC|<0.005 · `UNSTABLE_IC` tstat<1.0

# In[ ]:


# ── 3.2.5 Feature Quality Diagnostics ─────────────────────────────────────────
# Runs after extract_16_features, before walk-forward and quantum circuits.
# On quantum hardware: timestamps let you correlate feature diagnostics with
# the 1024-shot run log. PNG saved for persistent record across kernel restarts.

from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
# display replaced with print for script compatibility\n
# ── Thresholds ─────────────────────────────────────────────────────────────────
CORR_WARN    = 0.80   # |pairwise corr| above this  → HIGH_CORR
VIF_WARN     = 10.0   # VIF above this              → HIGH_VIF
IC_MIN       = 0.005  # |univariate IC| below this  → NO_IC (noise floor)
ICTSTAT_MIN  = 1.0    # IC t-stat below this        → UNSTABLE_IC
ROLL_WIN     = 60     # rolling IC window (trading days)

def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ── VIF ────────────────────────────────────────────────────────────────────────
def compute_vif(X_df):
    """Regress each feature on all others → VIF = 1 / (1 - R²).
    Catches collinearity that pairwise correlation misses (one feature ≈ combo of others)."""
    vifs = {}
    cols = X_df.columns.tolist()
    for col in cols:
        y_  = X_df[col].values
        X_  = X_df.drop(columns=[col]).values
        r2  = LinearRegression().fit(X_, y_).score(X_, y_)
        vifs[col] = 1.0 / (1.0 - r2 + 1e-12)
    return pd.Series(vifs, name="VIF")

# ── Per-bucket diagnostics ─────────────────────────────────────────────────────
def feature_diagnostics(feat_df, bucket_name):
    _log(f"{'='*60}")
    _log(f"Bucket: {bucket_name}")

    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols]
    y = feat_df["target"]
    n = len(feat_df)

    _log(f"  Rows after dropna : {n}")
    _log(f"  Features          : {len(feature_cols)}")

    # NaN audit — which feature is the binding constraint on dropna
    nan_per_feature = feat_df[feature_cols].isna().sum()
    if nan_per_feature.any():
        _log(f"  WARNING — NaN counts:\n{nan_per_feature[nan_per_feature > 0].to_string()}")
    else:
        _log("  NaN audit         : CLEAN (no mid-series NaNs)")

    # Distribution health
    _log("  Computing distribution stats...")
    stds   = X.std()
    skews  = X.skew().abs()

    # Univariate IC (full-sample)
    _log("  Computing univariate IC...")
    ics = X.apply(lambda col: col.corr(y))

    # Rolling IC → mean/std → t-stat
    roll_ic = X.apply(lambda col: col.rolling(ROLL_WIN).corr(y))
    ic_mean  = roll_ic.mean()
    ic_std   = roll_ic.std().replace(0, np.nan)
    ic_tstat = ic_mean / ic_std * np.sqrt(ROLL_WIN)

    # VIF
    _log("  Computing VIF (16 regressions)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vifs = compute_vif(X)

    # Max pairwise |r| per feature (excluding self)
    corr_mat     = X.corr()
    max_pair_corr = corr_mat.apply(lambda col: col.drop(col.name).abs().max())

    # Build table
    diag = pd.DataFrame({
        "Std Dev":       stds.round(4),
        "|Skew|":        skews.round(2),
        "IC":            ics.round(4),
        "IC t-stat":     ic_tstat.round(2),
        "Max |r|":       max_pair_corr.round(3),
        "VIF":           vifs.round(1),
    })

    flags = []
    for feat in feature_cols:
        f = []
        if max_pair_corr[feat]    > CORR_WARN:   f.append("HIGH_CORR")
        if vifs[feat]             > VIF_WARN:    f.append("HIGH_VIF")
        if abs(ics[feat])         < IC_MIN:      f.append("NO_IC")
        if abs(ic_tstat[feat])    < ICTSTAT_MIN: f.append("UNSTABLE_IC")
        flags.append(", ".join(f) if f else "OK")
    diag["Flag"] = flags

    _log(f"  Flagged: {sum(1 for f in flags if f != 'OK')}/{len(feature_cols)}  "
         f"| HIGH_CORR={sum('HIGH_CORR' in f for f in flags)} "
         f"| HIGH_VIF={sum('HIGH_VIF' in f for f in flags)} "
         f"| NO_IC={sum('NO_IC' in f for f in flags)} "
         f"| UNSTABLE_IC={sum('UNSTABLE_IC' in f for f in flags)}")

    print(
        diag.style
            .background_gradient(subset=["IC"],      cmap="RdYlGn", vmin=-0.08, vmax=0.08)
            .background_gradient(subset=["VIF"],     cmap="YlOrRd", vmin=1,     vmax=20)
            .background_gradient(subset=["Max |r|"], cmap="YlOrRd", vmin=0,     vmax=1)
            .map(lambda v: "color: crimson; font-weight: bold" if v != "OK" else
                           "color: #2d8a2d; font-weight: bold",  subset=["Flag"])
            .format({"IC": "{:+.4f}", "IC t-stat": "{:+.2f}", "VIF": "{:.1f}"})
            .set_caption(f"{bucket_name} — Feature Diagnostics")
    )

    return diag, corr_mat, roll_ic

# ── Run ────────────────────────────────────────────────────────────────────────
_log("Starting feature quality diagnostics...")
diag_A, corr_A, roll_ic_A = feature_diagnostics(feat_A, "Bucket A (AI/Tech)")
diag_B, corr_B, roll_ic_B = feature_diagnostics(feat_B, "Bucket B (Alt/Non-Core)")

# ── Plots ──────────────────────────────────────────────────────────────────────
_log("Rendering diagnostic plots...")

fig = plt.figure(figsize=(24, 16))
spec = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.30,
                         left=0.04, right=0.97, top=0.92, bottom=0.06)

axes = {
    "hm_A":  fig.add_subplot(spec[0, 0]),
    "ic_A":  fig.add_subplot(spec[0, 1:]),
    "hm_B":  fig.add_subplot(spec[1, 0]),
    "ic_B":  fig.add_subplot(spec[1, 1:]),
}

# Short labels for axes (strip long prefix/suffix)
def short_label(col):
    # e.g. "ret_20_BA_minus_SPY" → "ret_20"
    # e.g. "vol20_BA_minus_SPY"  → "vol20"
    return col.split("_minus_")[0].replace("_BA","").replace("_BB","")

def plot_heatmap(ax, corr_mat, title):
    mask  = np.triu(np.ones_like(corr_mat, dtype=bool))
    short = [short_label(c) for c in corr_mat.columns]
    cm_short = corr_mat.copy()
    cm_short.index   = short
    cm_short.columns = short
    sns.heatmap(
        cm_short, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 6.5},
        linewidths=0.4, square=True,
        cbar_kws={"shrink": 0.65, "label": "Pearson r"}
    )
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    # Gold border on cells exceeding threshold
    n = len(corr_mat)
    for i in range(n):
        for j in range(i):
            if abs(corr_mat.iloc[i, j]) > CORR_WARN:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False,
                    edgecolor="gold", lw=2.2, zorder=4))

def plot_rolling_ic(ax, roll_ic, diag, title):
    cmap  = plt.cm.tab20(np.linspace(0, 1, len(roll_ic.columns)))
    for i, feat in enumerate(roll_ic.columns):
        flagged = diag.loc[feat, "Flag"] != "OK"
        ax.plot(
            roll_ic.index, roll_ic[feat],
            label=short_label(feat),
            color=cmap[i],
            lw=1.5 if not flagged else 0.9,
            alpha=0.9 if not flagged else 0.35,
            ls="-"  if not flagged else "--"
        )
    ax.axhline(0,          color="black", lw=0.9, ls="-",  zorder=3)
    ax.axhline( IC_MIN,    color="seagreen", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(-IC_MIN,    color="seagreen", lw=0.8, ls=":", alpha=0.7,
               label=f"IC noise floor ±{IC_MIN}")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.set_ylabel(f"Rolling {ROLL_WIN}d IC", fontsize=8)
    ax.set_xlabel("Date", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, ncol=5, loc="upper left",
              framealpha=0.35, handlelength=1.2)
    ax.grid(True, alpha=0.20, lw=0.5)

plot_heatmap(axes["hm_A"], corr_A,
             "Bucket A — Correlation Matrix\n(gold border = |r| > 0.80)")
plot_rolling_ic(axes["ic_A"], roll_ic_A, diag_A,
                f"Bucket A — Rolling {ROLL_WIN}d IC per Feature\n(dashed = flagged)")

plot_heatmap(axes["hm_B"], corr_B,
             "Bucket B — Correlation Matrix\n(gold border = |r| > 0.80)")
plot_rolling_ic(axes["ic_B"], roll_ic_B, diag_B,
                f"Bucket B — Rolling {ROLL_WIN}d IC per Feature\n(dashed = flagged)")

fig.suptitle(
    f"Feature Quality Diagnostics  ·  "
    f"Thresholds: |r|>{CORR_WARN}  VIF>{VIF_WARN}  |IC|<{IC_MIN}  IC-tstat<{ICTSTAT_MIN}",
    fontsize=11, fontweight="bold"
)

plt.savefig("feature_diagnostics.png", dpi=150, bbox_inches="tight")
_log("Saved → feature_diagnostics.png")
plt.show()
_log("Feature diagnostics complete.")


# ### 3.3 MRU Quantum Circuit (N=15 → Q=4 qubits, r=4, D=2, M=10)
# 

# In[ ]:


# ── MRU Configuration ────────────────────────────────────────────────────
# N=15 features → Q=4 qubits (r=4 features/qubit), D=2 re-uploading layers, M=10 outputs
Q_MRU, N_FEAT, R_MRU, D_MRU, M_MRU = 4, 15, 4, 2, 10
# |W| = D * Q * R * 4 params/slot = 2*4*4*4 = 128
# (slot q3,k=3 is a bias pad: feat_idx=15 >= N_FEAT, so w1*x term is dropped)

# ── JAX backend setup ─────────────────────────────────────────────────────
# JAX vmap vectorizes the entire mini-batch into one compiled call (~30-100x vs serial loop).
# Falls back gracefully to serial NumPy if JAX is unavailable or on Braket.
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)   # match NumPy float64 precision
    _JAX_OK = True
    print("JAX available — MRU will use vectorized batch evaluation")
except ImportError:
    _JAX_OK = False
    print("JAX not found — falling back to serial NumPy evaluation (slower)")

if USE_BRAKET:
    dev_mru    = qml.device("braket.local.qubit", backend="default", wires=Q_MRU)
    _INTERFACE = "numpy"
elif _JAX_OK:
    dev_mru    = qml.device("default.qubit", wires=Q_MRU)
    _INTERFACE = "jax"
else:
    dev_mru    = qml.device("default.qubit", wires=Q_MRU)
    _INTERFACE = "numpy"

@qml.qnode(dev_mru, interface=_INTERFACE)
def mru_circuit(x, W):
    """
    Multiplexed Re-Uploading (MRU) circuit — paper Sec. 3.
    x : (N_FEAT,)  preprocessed feature vector in (-pi, pi)
    W : (D_MRU, Q_MRU, R_MRU, 4) — params [w0, w1, b, w2] per slot
    Returns all 15 Pauli-Z correlators (4 single + 6 two + 4 three + 1 four body).
    """
    for d in range(D_MRU):
        # Feature multiplexing: r ZYZ blocks per qubit
        for q in range(Q_MRU):
            for k in range(R_MRU):
                feat_idx = q * R_MRU + k
                w0, w1, b, w2 = W[d, q, k]
                qml.RZ(w0, wires=q)
                if feat_idx < N_FEAT:
                    qml.RY(w1 * x[feat_idx] + b, wires=q)   # data-dependent
                else:
                    qml.RY(b, wires=q)                       # bias pad (depth equalizer)
                qml.RZ(w2, wires=q)
        # Staggered entanglement (paper Eq. 5-6)
        if d == 0:   # E1: nearest-neighbor ring
            qml.CZ(wires=[0, 1]); qml.CZ(wires=[1, 2])
            qml.CZ(wires=[2, 3]); qml.CZ(wires=[3, 0])
        else:        # E2: cross-diagonal
            qml.CZ(wires=[0, 2]); qml.CZ(wires=[1, 3])
    # All 2^4 - 1 = 15 Pauli-Z observables
    single = [qml.expval(qml.PauliZ(i)) for i in range(Q_MRU)]
    two    = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
              for i in range(Q_MRU) for j in range(i+1, Q_MRU)]
    three  = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j) @ qml.PauliZ(k))
              for i in range(Q_MRU)
              for j in range(i+1, Q_MRU)
              for k in range(j+1, Q_MRU)]
    four   = [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))]
    return single + two + three + four   # 4+6+4+1 = 15

# ── Vectorized batch evaluator ────────────────────────────────────────────
# With JAX: jit+vmap processes the full batch in one compiled call.
# Without JAX: plain Python loop (same semantics, slower).
if _JAX_OK and not USE_BRAKET:
    _mru_batch_jax = jax.jit(jax.vmap(mru_circuit, in_axes=(0, None)))
    def _mru_batch(X, W):
        """Evaluate mru_circuit on all rows of X at once. Returns (N, 15) array."""
        out = _mru_batch_jax(jnp.array(X), jnp.array(W))
        if isinstance(out, (list, tuple)):
            out = jnp.stack(out, axis=1)  # PennyLane returns tuple of (batch,) per obs
        return np.array(out)
else:
    def _mru_batch(X, W):
        """Serial fallback — same output shape (N, 15)."""
        return np.array([mru_circuit(xi, W) for xi in X])


def init_W(seed=None):
    """Near-identity init (paper Eq. 12): w1=1, b=0, w0/w2 ~ N(0,0.01)."""
    rng = np.random.RandomState(seed)
    W = np.zeros((D_MRU, Q_MRU, R_MRU, 4))
    W[:, :, :, 1] = 1.0
    W[:, :, :, 0] = rng.randn(D_MRU, Q_MRU, R_MRU) * 0.01
    W[:, :, :, 3] = rng.randn(D_MRU, Q_MRU, R_MRU) * 0.01
    return W


def extract_mru_features(X, W, obs_idx):
    """Extract M_MRU features for all rows of X via batched circuit evaluation."""
    all_obs = _mru_batch(X, W)        # (N, 15) — one vectorized call
    return all_obs[:, obs_idx]


def spsa_train(W_init, X_tr, y_tr, n_steps=50, n_warmup=10,
               batch=16, a=0.05, c=0.1, A=10, ridge_alpha=1.0, seed=0):
    """
    Train MRU via SPSA (paper Sec. 6.3 — 2 circuit evals/step, param-count-independent).

    Phase 1 (n_warmup steps): warm up over all 15 observables.
    Phase 2: variance-rank observables on a subsample of X_tr, select top M_MRU
             (paper Eq. 9), continue with selected obs for remaining steps.

    SPSA schedule: ak = a/(k+1+A)^0.602, ck = c/(k+1)^0.101
    Returns (W_opt, obs_idx).
    """
    rng = np.random.RandomState(seed)
    W_flat   = W_init.ravel().copy()
    n_params = len(W_flat)
    obs_idx  = np.arange(15)          # use all during warmup

    def ridge_loss(W_f, o_idx):
        W_shaped = W_f.reshape(W_init.shape)
        F = extract_mru_features(X_batch, W_shaped, o_idx)   # uses _mru_batch internally
        A_mat = F.T @ F + ridge_alpha * np.eye(len(o_idx))
        beta  = np.linalg.solve(A_mat, F.T @ y_batch)
        r     = y_batch - F @ beta
        return float(r @ r) / len(y_batch)

    for step in range(n_steps):
        # Variance-ranked selection after warmup (paper Eq. 9)
        # Use at most 32 samples for selection to cap this one-time overhead.
        if step == n_warmup:
            W_shaped = W_flat.reshape(W_init.shape)
            n_var    = min(32, len(X_tr))
            var_samp = rng.choice(len(X_tr), size=n_var, replace=False)
            F_all    = extract_mru_features(X_tr[var_samp], W_shaped, np.arange(15))
            obs_idx  = np.argsort(F_all.var(axis=0))[::-1][:M_MRU]

        # Mini-batch
        idx = rng.choice(len(X_tr), size=min(batch, len(X_tr)), replace=False)
        X_batch, y_batch = X_tr[idx], y_tr[idx]   # closure vars

        # SPSA perturbation
        ak    = a / (step + 1 + A) ** 0.602
        ck    = c / (step + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=n_params)

        L_plus  = ridge_loss(W_flat + ck * delta, obs_idx)
        L_minus = ridge_loss(W_flat - ck * delta, obs_idx)
        grad    = (L_plus - L_minus) / (2 * ck) * delta
        W_flat -= ak * grad

    # Final obs selection if n_steps <= n_warmup
    if n_steps <= n_warmup:
        W_shaped = W_flat.reshape(W_init.shape)
        n_var    = min(32, len(X_tr))
        var_samp = rng.choice(len(X_tr), size=n_var, replace=False)
        F_all    = extract_mru_features(X_tr[var_samp], W_shaped, np.arange(15))
        obs_idx  = np.argsort(F_all.var(axis=0))[::-1][:M_MRU]

    return W_flat.reshape(W_init.shape), obs_idx


# ── Smoke test ────────────────────────────────────────────────────────────
_W0  = init_W(seed=0)
_x0  = np.zeros(N_FEAT)
_out = _mru_batch(np.array([_x0]), _W0)[0]   # test vectorized call

# Use a plain NumPy circuit for qml.specs (avoids JAX tracing in specs)
_dev_spec = qml.device("default.qubit", wires=Q_MRU)
@qml.qnode(_dev_spec, interface="numpy")
def _mru_for_specs(x, W):
    for d in range(D_MRU):
        for q in range(Q_MRU):
            for k in range(R_MRU):
                feat_idx = q * R_MRU + k
                w0, w1, b, w2 = W[d, q, k]
                qml.RZ(w0, wires=q)
                if feat_idx < N_FEAT:
                    qml.RY(w1 * x[feat_idx] + b, wires=q)
                else:
                    qml.RY(b, wires=q)
                qml.RZ(w2, wires=q)
        if d == 0:
            qml.CZ(wires=[0,1]); qml.CZ(wires=[1,2])
            qml.CZ(wires=[2,3]); qml.CZ(wires=[3,0])
        else:
            qml.CZ(wires=[0,2]); qml.CZ(wires=[1,3])
    return [qml.expval(qml.PauliZ(i)) for i in range(Q_MRU)]

_specs = qml.specs(_mru_for_specs)(_x0, _W0)["resources"]
print(f"MRU circuit: qubits={_specs.num_wires}  depth={_specs.depth}  "
      f"gates={_specs.num_gates}  params={_W0.size}  all_obs={len(_out)}  selected_M={M_MRU}")
print(f"Batch evaluator: {'JAX jit+vmap' if (_JAX_OK and not USE_BRAKET) else 'NumPy serial'}")


# ### 3.4 Walk-Forward Backtest

# In[ ]:


# ── 4. Walk-Forward Backtest — Classical | Tanh Control | Real MRU ──────────
# Requires: mru_circuit, init_W, spsa_train, extract_mru_features (defined in MRU section above)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
# ── Runtime knobs ──────────────────────────────────────────────────────────
DEBUG_MODE      = False   # True = first 20 windows only; False = all windows
N_WINDOWS_DEBUG = 20     # windows to run in debug mode
SPSA_STEPS      = 25     # steps in debug; bump to 50 for full run
SPSA_WARMUP     = 8
SPSA_BATCH      = 16
SPSA_A          = 0.05
SPSA_C          = 0.1
SPSA_A_OFFSET   = 10
RIDGE_ALPHA_MRU = 1.0
RETRAIN_EVERY_K = 5      # retrain MRU every K windows (1=always; 5=every 5th)
LOOKBACK_WINDOW = 104

poly2_wf = PolynomialFeatures(degree=2, include_bias=False)

buckets = {"Bucket A (AI/Tech)": feat_A, "Bucket B (Alt/Non-Core)": feat_B}
all_results = []

print(f"=== Walk-Forward Backtest ===")
print(f"  DEBUG_MODE={DEBUG_MODE}  N_WINDOWS_DEBUG={N_WINDOWS_DEBUG}  SPSA_STEPS={SPSA_STEPS}")
print(f"  RETRAIN_EVERY_K={RETRAIN_EVERY_K}  LOOKBACK={LOOKBACK_WINDOW}")

for bucket_name, df in buckets.items():
    print(f"\n--- {bucket_name} ---")

    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols].values
    y = df['target'].values

    total_windows = len(df) - LOOKBACK_WINDOW
    window_range  = range(LOOKBACK_WINDOW, len(df))
    if DEBUG_MODE:
        window_range = range(LOOKBACK_WINDOW, LOOKBACK_WINDOW + min(N_WINDOWS_DEBUG, total_windows))

    actuals            = []
    preds_classical    = []
    preds_tanh_control = []
    preds_mru          = []

    # Warm-start state
    W_prev   = None
    obs_prev = None

    for loop_i, i in enumerate(window_range):
        # ── A. Split ──────────────────────────────────────────────────────
        X_train_raw   = X[i - LOOKBACK_WINDOW : i - 5]
        y_train_win   = y[i - LOOKBACK_WINDOW : i - 5]
        X_test_raw    = X[i : i + 1]
        y_test_actual = y[i]

        # ── B. Classical preprocessing ────────────────────────────────────
        X_tr_clipped = np.clip(X_train_raw, -5, 5)
        X_te_clipped = np.clip(X_test_raw,  -5, 5)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_clipped)
        X_te_s = scaler.transform(X_te_clipped)

        # ── C. Quantum input branch (bounded) ─────────────────────────────
        X_tr_q = np.pi * np.tanh(X_tr_s / 2)
        X_te_q = np.pi * np.tanh(X_te_s / 2)

        # ── D. Classical engineered features (poly2 on standard-scaled) ───
        X_tr_p2 = poly2_wf.fit_transform(X_tr_s)
        X_te_p2 = poly2_wf.transform(X_te_s)

        # ── Model 1: Classical Ridge Poly2 ────────────────────────────────
        ridge_cl = Ridge(alpha=RIDGE_ALPHA_MRU).fit(X_tr_p2, y_train_win)
        preds_classical.append(ridge_cl.predict(X_te_p2)[0])

        # ── Model 2: Tanh Control (poly2 on tanh-squashed inputs) ─────────
        X_tr_p2_q = poly2_wf.fit_transform(X_tr_q)
        X_te_p2_q = poly2_wf.transform(X_te_q)
        ridge_tc  = Ridge(alpha=RIDGE_ALPHA_MRU).fit(X_tr_p2_q, y_train_win)
        preds_tanh_control.append(ridge_tc.predict(X_te_p2_q)[0])

        # ── Model 3: Classical + Real MRU ─────────────────────────────────
        win_seed = 42 + loop_i  # deterministic per window

        retrain = (loop_i % RETRAIN_EVERY_K == 0)
        if retrain:
            W_init = W_prev if W_prev is not None else init_W(seed=win_seed)
            W_opt, obs_idx = spsa_train(
                W_init, X_tr_q, y_train_win,
                n_steps=SPSA_STEPS,
                n_warmup=SPSA_WARMUP,
                batch=min(SPSA_BATCH, len(X_tr_q)),
                a=SPSA_A, c=SPSA_C, A=SPSA_A_OFFSET,
                ridge_alpha=RIDGE_ALPHA_MRU,
                seed=win_seed,
            )
            W_prev   = W_opt
            obs_prev = obs_idx
        else:
            W_opt    = W_prev
            obs_idx  = obs_prev

        Q_tr_mru = extract_mru_features(X_tr_q, W_opt, obs_idx)
        Q_te_mru = extract_mru_features(X_te_q, W_opt, obs_idx)

        # Concatenate classical poly2 + MRU, then re-standardize jointly
        Xaug_tr = np.hstack([X_tr_p2, Q_tr_mru])
        Xaug_te = np.hstack([X_te_p2, Q_te_mru])
        sc_aug  = StandardScaler()
        Xaug_tr = sc_aug.fit_transform(Xaug_tr)
        Xaug_te = sc_aug.transform(Xaug_te)

        ridge_mru = Ridge(alpha=RIDGE_ALPHA_MRU).fit(Xaug_tr, y_train_win)
        preds_mru.append(ridge_mru.predict(Xaug_te)[0])

        actuals.append(y_test_actual)

        # ── Debug assertions & prints (first 2 windows) ───────────────────
        if loop_i < 2:
            assert not np.any(np.isnan(X_tr_q)),   f"NaN in X_tr_q at window {loop_i}"
            assert not np.any(np.isnan(Q_tr_mru)),  f"NaN in Q_tr_mru at window {loop_i}"
            assert not np.any(np.isnan(Xaug_tr)),   f"NaN in Xaug_tr at window {loop_i}"
            assert Q_tr_mru.shape[1] == len(obs_idx), "obs_idx size mismatch"
            assert X_te_q.shape[0] == 1,             "test sample must be exactly one row"
            print(f"  [win {loop_i}] X_tr_p2={X_tr_p2.shape}  Q_tr_mru={Q_tr_mru.shape}  "
                  f"obs_idx={obs_idx}  Q_std_min={Q_tr_mru.std(axis=0).min():.4f}")

        if loop_i % max(1, len(list(window_range)) // 5) == 0:
            print(f"  window {loop_i}/{len(list(window_range))} done", flush=True)

    # ── Per-bucket metrics for all three models ───────────────────────────
    act  = np.array(actuals)
    def bucket_row(name, preds):
        p = np.array(preds)
        ic_p = np.corrcoef(act, p)[0, 1] if len(act) > 1 else float('nan')
        ic_s = spearmanr(act, p).correlation if len(act) > 1 else float('nan')
        return {
            "Bucket":          bucket_name,
            "Model":           name,
            "MSE":             mean_squared_error(act, p),
            "MAE":             mean_absolute_error(act, p),
            "Pearson IC":      ic_p,
            "Spearman IC":     ic_s,
            "N_windows":       len(act),
        }

    all_results.append(bucket_row("Classical Ridge Poly2",    preds_classical))
    all_results.append(bucket_row("Ridge Poly2 Tanh Control", preds_tanh_control))
    all_results.append(bucket_row("Ridge Poly2 + MRU",        preds_mru))

# ── Aggregate display ─────────────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
print("\n=== Walk-Forward Results (All Models) ===")
print(df_results.sort_values(["Bucket", "Pearson IC"], ascending=[True, False]))

# ── Uplift tables ─────────────────────────────────────────────────────────
uplift_rows = []
for bname, grp in df_results.groupby("Bucket"):
    cl  = grp[grp["Model"] == "Classical Ridge Poly2"]["Pearson IC"].values[0]
    tc  = grp[grp["Model"] == "Ridge Poly2 Tanh Control"]["Pearson IC"].values[0]
    mru = grp[grp["Model"] == "Ridge Poly2 + MRU"]["Pearson IC"].values[0]
    uplift_rows.append({"Bucket": bname, "MRU - Classical": mru - cl, "Tanh - Classical": tc - cl})

df_uplift = pd.DataFrame(uplift_rows)
print("\n=== IC Uplift vs Classical Baseline ===")
print(df_uplift)

# ── Build wf_df for plotting ──────────────────────────────────────────────
cl_rows  = df_results[df_results["Model"] == "Classical Ridge Poly2"].reset_index(drop=True)
tc_rows  = df_results[df_results["Model"] == "Ridge Poly2 Tanh Control"].reset_index(drop=True)
mru_rows = df_results[df_results["Model"] == "Ridge Poly2 + MRU"].reset_index(drop=True)

wf_df = pd.DataFrame({
    "Bucket":          cl_rows["Bucket"],
    "CL_Corr":         cl_rows["Pearson IC"],
    "TC_Corr":         tc_rows["Pearson IC"],
    "MRU_Corr":        mru_rows["Pearson IC"],
})
wf_df["MRU_\u0394Corr"] = wf_df["MRU_Corr"] - wf_df["CL_Corr"]
wf_df["TC_\u0394Corr"]  = wf_df["TC_Corr"]  - wf_df["CL_Corr"]
print("\n", wf_df.to_string(index=False))


# ### 3.5 Walk-Forward Results — Visualisation

# In[ ]:


# wf_df is built directly in the walk-forward loop cell above.
# This cell is a no-op kept to avoid renumbering downstream cells.
pass


# In[ ]:


# ── Visualization: IC comparison + uplift bars ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(len(wf_df))
w = 0.25
labels = wf_df["Bucket"].str.extract(r'Bucket (\w+)')[0].values

axes[0].bar(x_pos - w,   wf_df["CL_Corr"],  w, label="Classical Ridge Poly2",    color="#4878CF", alpha=0.85)
axes[0].bar(x_pos,       wf_df["TC_Corr"],  w, label="Ridge Poly2 Tanh Control",  color="#6ACC65", alpha=0.85)
axes[0].bar(x_pos + w,   wf_df["MRU_Corr"], w, label="Ridge Poly2 + MRU",         color="#E24A33", alpha=0.85)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels, rotation=45)
axes[0].axhline(0, c="k", lw=0.7)
axes[0].set_ylabel("IC (Pearson Corr)")
axes[0].set_title("Walk-Forward IC: Classical vs Tanh Control vs Real MRU")
axes[0].legend(fontsize=8)

axes[1].bar(x_pos - w/2, wf_df["MRU_\u0394Corr"], w, label="MRU \u2013 Classical",         color="#E24A33", alpha=0.85)
axes[1].bar(x_pos + w/2, wf_df["TC_\u0394Corr"],  w, label="Tanh Control \u2013 Classical", color="#6ACC65", alpha=0.85)
axes[1].axhline(0, c="k", lw=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels, rotation=45)
axes[1].set_ylabel("\u0394IC vs Classical Ridge")
axes[1].set_title("Uplift: MRU vs Tanh-Control Preprocessing Only")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("part2_results.png", bbox_inches='tight')
plt.show()

print("\n=== Interpretation ===")
print("  'Tanh Control \u2013 Classical' isolates the effect of bounded preprocessing alone.")
print("  'MRU \u2013 Classical' is the full quantum feature augmentation effect.")
print("  If MRU uplift >> Tanh uplift: quantum observables are contributing signal.")
print("  If MRU uplift \u2248 Tanh uplift: benefit comes from preprocessing, not circuit.")


# ---
# ## 4. Quantum Resource Usage

# In[ ]:


print("=== Quantum Resource Summary ===\n")

print("Part I — 4-qubit circuits:")
for name, circ, x_in in [("Angle+ZZ", angle_circuit, X_train_q[0])]:
    s = qml.specs(circ)(x_in)["resources"]
    print(f"  {name:20s}  qubits={s.num_wires}  depth={s.depth:3d}  "
          f"gates={s.num_gates:3d}  output_features=10")

_x_spec = np.zeros(N_FEAT)
_W_spec = init_W(seed=0)
s2 = qml.specs(mru_circuit)(_x_spec, _W_spec)["resources"]
print(f"\nPart II — MRU circuit (N=15→Q=4, r=4, D=2, M=10):")
print(f"  {'MRU':20s}  qubits={s2.num_wires}  depth={s2.depth:3d}  "
      f"gates={s2.num_gates:3d}  params={_W_spec.size}  output_features={M_MRU}")
print(f"  SPSA: 2 circuit evals/step × 50 steps/window  (param-count-independent)")

# ── Cost–Performance ──────────────────────────────────────────────────────
best_q_poly_mse   = min(r["MSE"] for r in results_p1 if "Poly2+" in r["Model"] and r["MSE"])
best_cl_poly2_mse = next(r["MSE"] for r in results_p1 if "poly deg=2" in r["Model"])
best_cl_poly3_mse = next(r["MSE"] for r in results_p1 if "poly deg=3" in r["Model"])

print(f"\n=== Cost–Performance Analysis ===")
print(f"  Best Q+Poly2 MSE (Part I): {best_q_poly_mse:.4f}")
print(f"  Classical poly2 MSE:       {best_cl_poly2_mse:.4f}  "
      f"(Δ = {(best_cl_poly2_mse-best_q_poly_mse)/best_cl_poly2_mse*100:+.1f}%)")
print(f"  Classical poly3 MSE:       {best_cl_poly3_mse:.4f}")
print(f"  Part II — Mean IC: Classical={wf_df['CL_Corr'].mean():+.4f}  "
      f"MRU={wf_df['MRU_Corr'].mean():+.4f}  (Δ={wf_df['MRU_ΔCorr'].mean():+.4f})")
print(f"  Stocks with positive MRU ΔIC: {(wf_df['MRU_ΔCorr'] > 0).sum()}/10")
# wf_df['t_s'] removed — timing not tracked in bucket-level backtest


# ---
# ## 5. Discussion & Conclusions
# 
# ### Part I — Synthetic Regime Switching
# 
# | Finding | Detail |
# |---|---|
# | **Quantum alone underperforms poly3** | Angle+ZZ ridge gives MSE=3.26 vs poly3 MSE=1.22 |
# | **Quantum adds value over poly2** | Ridge+Poly2+Angle: MSE=1.445 vs Ridge+Poly2: MSE=1.627 (−11%) |
# | **IQP and ZZ provide modest uplift** | When combined with poly2, all quantum maps improve over raw poly2 |
# | **Key driver** | ZZ cross-feature encoding captures $X_1 \cdot X_3$ interactions driving Regime 2 |
# 
# ### Part II — Walk-Forward Backtest (S&P 500)
# 
# | Metric | Classical Ridge | MRU (trained VQC) |
# |---|---|---|
# | Mean IC | see output | see output |
# | Stocks with positive ΔIC | — | see output |
# 
# **MRU architecture (paper: Multiplexed Re-Uploading Feature Map):**
# - 4 qubits encode 15 features via multiplexing (r=4 features/qubit, 4 ZYZ blocks/qubit/layer)
# - D=2 re-uploading layers with staggered CZ entanglement (ring E1, cross-diagonal E2)
# - Variance-ranked Pauli-Z observable selection (top M=10 from 15 available)
# - SPSA end-to-end training: 2 circuit evals/step, 50 steps/window, per-window re-initialization
# - Near-identity init prevents barren plateaus
# 
# **Key observations (fill in after running):**
# - MRU uses 4× fewer qubits than the previous 15-qubit static circuit
# - Trainable parameters allow the circuit to adapt to each stock's feature distribution per window
# - 3-body and 4-body Pauli-Z correlators provide richer nonlinear feature interactions than angle encoding
# 
# ### Quantum Resource Tradeoffs
# 
# | Circuit | Qubits | Depth | Output | Trainable Params | Hardware |
# |---|---|---|---|---|---|
# | Angle+ZZ (Part I) | 4 | 2 | 10 | 0 | Any NISQ |
# | ZZ Feature Map (Part I) | 4 | 31 | 10 | 0 | Deep NISQ |
# | IQP Encoding (Part I) | 4 | 1 | 10 | 0 | Any NISQ |
# | **MRU (Part II)** | **4** | **~12** | **10** | **128** | **IonQ Aria-1** |
# 
# ### Honest Conclusion
# 
# The MRU architecture replaces the static 15-qubit angle circuit with a hardware-efficient 4-qubit VQC
# that trains end-to-end via SPSA. Whether it outperforms the classical Ridge baseline depends on whether
# the SPSA-trained variational parameters discover feature interactions beyond what linear Ridge captures —
# both positive and negative outcomes are scientifically valid under this rigorous methodology.
# 
# Future directions:
# - Increase SPSA steps (100–200) for better convergence
# - Warm-start parameters from previous window (reduce per-window training cost)
# - Probability-basis extraction (2^4 = 16 amplitudes) instead of expectation values
# - QPU deployment on IonQ Aria-1 (circuit depth within decoherence budget: r_max ≈ 6 for T2≈10s)
# 

# In[1]:


print("=== Final Summary ===")
print(f"\nPart I (Synthetic) — Best models:")
for r in sorted(results_p1, key=lambda x: x["MSE"])[:4]:
    print(f"  {r['Model']:40s}  MSE={r['MSE']:.4f}  Corr={r['Corr']:.4f}")
print(f"\nPart II (S&P 500) — Aggregate IC:")
print(f"  Classical Ridge mean IC: {wf_df['CL_Corr'].mean():+.4f}")
print(f"  MRU-augmented mean IC:   {wf_df['MRU_Corr'].mean():+.4f}  (Δ={wf_df['MRU_ΔCorr'].mean():+.4f})")
print(f"  Stocks positive ΔIC:     {(wf_df['MRU_ΔCorr'] > 0).sum()}/10")
print("\n✓ Notebook complete — ready for AWS submission")


# In[ ]:




