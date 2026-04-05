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
    
    for tic, w in weights.items():
        # Sum weighted returns
        ret_i = df_raw["Close"][tic].pct_change()
        bucket_ret += w * ret_i.fillna(0)
        # Sum weighted dollar volume
        bucket_dv += w * (df_raw["Close"][tic] * df_raw["Volume"][tic])
        
    # Reconstruct Close price starting at 100
    bucket_close = [100.0]
    for r in bucket_ret.iloc[1:]:
        bucket_close.append(bucket_close[-1] * (1 + r))
        
    bucket_df = pd.DataFrame(index=idx)
    bucket_df["Close"] = bucket_close
    bucket_df["Dollar_Volume"] = bucket_dv
    return bucket_df

print("Building Synthetic Buckets...")
bA_df = build_bucket(raw, w_A)
bB_df = build_bucket(raw, w_B)
print("Buckets constructed successfully.")


# ── 3.2 Feature Engineering (Exactly 16 Features per Bucket) ───────────

def extract_16_features(bucket_df, proxies_df, name, proxy1, proxy2):
    f = pd.DataFrame(index=bucket_df.index)
    bc = bucket_df["Close"]
    bdv = bucket_df["Dollar_Volume"]
    spy_c = proxies_df["Close"]["SPY"]
    p1_c = proxies_df["Close"][proxy1]
    p2_c = proxies_df["Close"][proxy2]
    
    # helper for n-day return
    def ret(series, n): return series.pct_change(n)
    
    # Q1: Relative Momentum
    f[f"ret_5_{name}_minus_SPY"] = ret(bc, 5) - ret(spy_c, 5)
    f[f"ret_20_{name}_minus_SPY"] = ret(bc, 20) - ret(spy_c, 20)
    f[f"ret_60_{name}_minus_SPY"] = ret(bc, 60) - ret(spy_c, 60)
    f[f"ret_100_{name}_minus_SPY"] = ret(bc, 100) - ret(spy_c, 100)
    
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

    # 5-day volume-weighted return — price × vol interaction; README-listed
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


# ── 4. Walk-Forward Backtest & Evaluation (Bucket Level) ───────────────
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from IPython.display import display

LOOKBACK_WINDOW = 504 
poly2 = PolynomialFeatures(degree=2, include_bias=False)

# Group our new buckets together
buckets = {"Bucket A (AI/Tech)": feat_A, "Bucket B (Alt/Non-Core)": feat_B}
all_results = []

print(f"Starting Walk-Forward Backtest (Window: {LOOKBACK_WINDOW} days)...")

for bucket_name, df in buckets.items():
    print(f"Processing {bucket_name}...")
    
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols].values
    y = df['target'].values
    
    # Storage for this bucket's predictions
    actuals, preds_ridge_s, preds_ridge_q = [], [], []
    
    # Walk-forward loop
    for i in range(LOOKBACK_WINDOW, len(df)):
        # 1. Split Train/Test with the CRITICAL 5-DAY GAP to prevent lookahead bias
        X_train_raw = X[i - LOOKBACK_WINDOW : i - 5]
        y_train = y[i - LOOKBACK_WINDOW : i - 5]
        
        X_test_raw = X[i : i+1]
        y_test_actual = y[i]
        
        # 2. Preprocessing Phase 1: Cauchy Clip -> Standard Scale
        # Clip train to prevent Cauchy contamination
        X_tr_clipped = np.clip(X_train_raw, -5, 5)
        X_te_clipped = np.clip(X_test_raw, -5, 5)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_clipped)
        X_te_s = scaler.transform(X_te_clipped)
        
        # 3. Preprocessing Phase 2: Tanh Squash (for Quantum baseline)
        X_tr_q = np.pi * np.tanh(X_tr_s / 2)
        X_te_q = np.pi * np.tanh(X_te_s / 2)
        
        # 4. Polynomial Expansion (Degree 2)
        X_tr_p2_s = poly2.fit_transform(X_tr_s)
        X_te_p2_s = poly2.transform(X_te_s)
        
        X_tr_p2_q = poly2.fit_transform(X_tr_q)
        X_te_p2_q = poly2.transform(X_te_q)
        
        # 5. Train & Predict Optimal Classical (Standard Data)
        ridge_s = Ridge(alpha=1.0).fit(X_tr_p2_s, y_train)
        preds_ridge_s.append(ridge_s.predict(X_te_p2_s)[0])
        
        # 6. Train & Predict Control Classical (Tanh Data)
        ridge_q = Ridge(alpha=1.0).fit(X_tr_p2_q, y_train)
        preds_ridge_q.append(ridge_q.predict(X_te_p2_q)[0])
        
        actuals.append(y_test_actual)
        
    # Evaluate Bucket
    act = np.array(actuals)
    
    all_results.append({
        "Bucket": bucket_name,
        "Model": "Ridge Poly2 (Optimal)",
        "MSE": mean_squared_error(act, preds_ridge_s),
        "MAE": mean_absolute_error(act, preds_ridge_s),
        "Corr (IC)": np.corrcoef(act, preds_ridge_s)[0, 1]
    })
    
    all_results.append({
        "Bucket": bucket_name,
        "Model": "Ridge Poly2 (Tanh Control)",
        "MSE": mean_squared_error(act, preds_ridge_q),
        "MAE": mean_absolute_error(act, preds_ridge_q),
        "Corr (IC)": np.corrcoef(act, preds_ridge_q)[0, 1]
    })

# Display Final Leaderboard
df_results = pd.DataFrame(all_results)
print("\n=== Final Classical Baselines (Bucket Level) ===")
display(df_results.sort_values(by=["Bucket", "MSE"]))