import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("quantum_outputs")
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SPEC_TRAIN_PATH = DATA_DIR / "part1_spec_train.csv"
SPEC_TEST_PATH  = DATA_DIR / "part1_spec_test.csv"
MRU_FEAT_PATH   = DATA_DIR / "part2_mru_features.csv"
MRU_PRED_PATH   = DATA_DIR / "part2_mru_predictions.csv"

# ============================================================
# LOAD
# ============================================================
spec_train = pd.read_csv(SPEC_TRAIN_PATH)
spec_test  = pd.read_csv(SPEC_TEST_PATH)

mru_feat = pd.read_csv(MRU_FEAT_PATH)
mru_pred = pd.read_csv(MRU_PRED_PATH)

mru_feat["date"] = pd.to_datetime(mru_feat["date"])
mru_pred["date"] = pd.to_datetime(mru_pred["date"])

spec_cols = [c for c in spec_test.columns if c.startswith("spec_f")]
mru_cols  = [c for c in mru_feat.columns if c.startswith("mru_f")]

print("Loaded:")
print(" spec_train:", spec_train.shape)
print(" spec_test :", spec_test.shape)
print(" mru_feat  :", mru_feat.shape)
print(" mru_pred  :", mru_pred.shape)

# ============================================================
# HELPERS
# ============================================================
def savefig(name):
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved -> {path}")

def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def rolling_corr(x, y, window=60):
    x = pd.Series(x)
    y = pd.Series(y)
    return x.rolling(window).corr(y)

def plot_heatmap_from_corr(corr_df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, aspect="auto")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    savefig(filename)

# ============================================================
# PART 1: SPEC FEATURE MAP PLOTS
# ============================================================

# 1) Feature-vs-target correlation bars (train/test)
train_corrs = [safe_corr(spec_train[c], spec_train["y"]) for c in spec_cols]
test_corrs  = [safe_corr(spec_test[c],  spec_test["y"])  for c in spec_cols]

x = np.arange(len(spec_cols))
w = 0.38

plt.figure(figsize=(8, 4.5))
plt.bar(x - w/2, train_corrs, width=w, label="train")
plt.bar(x + w/2, test_corrs,  width=w, label="test")
plt.axhline(0, linewidth=1)
plt.xticks(x, spec_cols)
plt.ylabel("Pearson corr with y")
plt.title("Part 1 SPEC features: correlation with target")
plt.legend()
savefig("part1_spec_feature_target_corr.png")

# 2) Regime-conditional feature-vs-target correlations on test
regimes = sorted(spec_test["regime"].unique())
corr_mat = pd.DataFrame(index=[f"regime_{r}" for r in regimes], columns=spec_cols, dtype=float)
for r in regimes:
    df_r = spec_test[spec_test["regime"] == r]
    for c in spec_cols:
        corr_mat.loc[f"regime_{r}", c] = safe_corr(df_r[c], df_r["y"])

plot_heatmap_from_corr(
    corr_mat,
    "Part 1 SPEC test: feature-target correlation by regime",
    "part1_spec_regime_feature_target_heatmap.png"
)

# 3) Feature correlation matrix on test
spec_corr = spec_test[spec_cols + ["y"]].corr()
plot_heatmap_from_corr(
    spec_corr,
    "Part 1 SPEC test: feature correlation matrix",
    "part1_spec_feature_corr_matrix.png"
)

# 4) Histograms by regime for each feature
for c in spec_cols:
    plt.figure(figsize=(7, 4))
    for r in regimes:
        vals = spec_test.loc[spec_test["regime"] == r, c].dropna()
        plt.hist(vals, bins=50, alpha=0.5, density=True, label=f"regime {r}")
    plt.title(f"{c} distribution by regime (test)")
    plt.xlabel(c)
    plt.ylabel("Density")
    plt.legend()
    savefig(f"part1_{c}_hist_by_regime.png")

# 5) Scatter plots of top 2 most predictive test features vs y
abs_test_corr_order = np.argsort(np.abs(test_corrs))[::-1]
top2 = [spec_cols[i] for i in abs_test_corr_order[:2]]

for c in top2:
    plt.figure(figsize=(5.5, 4.5))
    for r in regimes:
        df_r = spec_test[spec_test["regime"] == r]
        plt.scatter(df_r[c], df_r["y"], s=8, alpha=0.35, label=f"regime {r}")
    plt.xlabel(c)
    plt.ylabel("y")
    plt.title(f"Part 1 test: {c} vs y")
    plt.legend()
    savefig(f"part1_{c}_vs_y_scatter.png")

# ============================================================
# PART 2: MRU PREDICTION PLOTS
# ============================================================

for bucket in sorted(mru_pred["bucket"].unique()):
    df = mru_pred[mru_pred["bucket"] == bucket].sort_values("date").copy()
    df["resid"] = df["actual"] - df["y_hat_mru"]
    df["rolling_ic_60"] = rolling_corr(df["y_hat_mru"], df["actual"], window=60)

    # 1) Actual vs predicted through time
    plt.figure(figsize=(12, 4.8))
    plt.plot(df["date"], df["actual"], label="actual")
    plt.plot(df["date"], df["y_hat_mru"], label="predicted")
    retrain_dates = df.loc[df["retrained"] == 1, "date"]
    ymin = min(df["actual"].min(), df["y_hat_mru"].min())
    ymax = max(df["actual"].max(), df["y_hat_mru"].max())
    for d in retrain_dates:
        plt.vlines(d, ymin, ymax, linestyles="dashed", alpha=0.2)
    plt.title(f"{bucket}: actual vs predicted")
    plt.xlabel("date")
    plt.ylabel("5d excess return")
    plt.legend()
    savefig(f"{bucket.lower()}_actual_vs_predicted.png")

    # 2) Scatter predicted vs actual
    rho = safe_corr(df["y_hat_mru"], df["actual"])
    plt.figure(figsize=(5.5, 5))
    plt.scatter(df["y_hat_mru"], df["actual"], s=10, alpha=0.45)
    lo = min(df["y_hat_mru"].min(), df["actual"].min())
    hi = max(df["y_hat_mru"].max(), df["actual"].max())
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title(f"{bucket}: predicted vs actual (corr={rho:.3f})")
    savefig(f"{bucket.lower()}_predicted_vs_actual_scatter.png")

    # 3) Residual histogram
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(df["resid"], bins=50, alpha=0.8)
    plt.axvline(0, linewidth=1)
    plt.title(f"{bucket}: residual distribution")
    plt.xlabel("actual - predicted")
    plt.ylabel("count")
    savefig(f"{bucket.lower()}_residual_hist.png")

    # 4) Rolling IC
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["rolling_ic_60"])
    plt.axhline(0, linewidth=1)
    plt.title(f"{bucket}: 60-day rolling correlation(predicted, actual)")
    plt.xlabel("date")
    plt.ylabel("rolling IC")
    savefig(f"{bucket.lower()}_rolling_ic_60.png")

# ============================================================
# PART 2: MRU FEATURE PLOTS
# ============================================================

for bucket in sorted(mru_feat["bucket"].unique()):
    df = mru_feat[mru_feat["bucket"] == bucket].sort_values("date").copy()

    # 1) Feature-target correlations
    feat_corrs = [safe_corr(df[c], df["actual"]) for c in mru_cols]

    plt.figure(figsize=(9, 4.2))
    plt.bar(np.arange(len(mru_cols)), feat_corrs)
    plt.axhline(0, linewidth=1)
    plt.xticks(np.arange(len(mru_cols)), mru_cols, rotation=45, ha="right")
    plt.ylabel("Pearson corr with actual")
    plt.title(f"{bucket}: MRU feature-target correlations")
    savefig(f"{bucket.lower()}_mru_feature_target_corr.png")

    # 2) Feature correlation matrix
    corr_df = df[mru_cols + ["actual"]].corr()
    plot_heatmap_from_corr(
        corr_df,
        f"{bucket}: MRU feature correlation matrix",
        f"{bucket.lower()}_mru_feature_corr_matrix.png"
    )

    # 3) Top 3 feature time series by absolute corr with actual
    top3_idx = np.argsort(np.abs(feat_corrs))[::-1][:3]
    top3 = [mru_cols[i] for i in top3_idx]

    plt.figure(figsize=(12, 5))
    for c in top3:
        plt.plot(df["date"], df[c], label=c)
    plt.title(f"{bucket}: top 3 MRU features over time")
    plt.xlabel("date")
    plt.ylabel("feature value")
    plt.legend()
    savefig(f"{bucket.lower()}_top3_mru_features_timeseries.png")

# ============================================================
# SUMMARY TABLES
# ============================================================
summary_rows = []

# Part 1
for c in spec_cols:
    summary_rows.append({
        "section": "part1_spec",
        "name": c,
        "train_corr_y": safe_corr(spec_train[c], spec_train["y"]),
        "test_corr_y": safe_corr(spec_test[c], spec_test["y"]),
    })

# Part 2 predictions
for bucket in sorted(mru_pred["bucket"].unique()):
    df = mru_pred[mru_pred["bucket"] == bucket].sort_values("date").copy()
    resid = df["actual"] - df["y_hat_mru"]
    summary_rows.append({
        "section": "part2_pred",
        "name": bucket,
        "train_corr_y": np.nan,
        "test_corr_y": safe_corr(df["y_hat_mru"], df["actual"]),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = PLOT_DIR / "plot_summary_metrics.csv"
summary_df.to_csv(summary_path, index=False)
print(f"saved -> {summary_path}")

print("\nDone. Plots written to:", PLOT_DIR.resolve())