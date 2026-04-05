# ── Imports ───────────────────────────────────────────────────────────────────
import pennylane as qml
import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy.stats import cauchy

# ── Hardware flag ─────────────────────────────────────────────────────────────
USE_BRAKET = False          # ← Set True for real QPU execution
QPU_ARN    = "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"  # ← fill in your QPU
S3_BUCKET  = "amazon-braket-your-bucket"   # ← fill in your S3 bucket
S3_PREFIX  = "qfa-quantum-features"
SHOTS      = 1024           # used for QPU; ignored by statevec simulator

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("quantum_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Part I device (4 qubits) ──────────────────────────────────────────────────
Q_P1 = 4
if USE_BRAKET:
    dev_p1 = qml.device(
        "braket.aws.qubit", device_arn=QPU_ARN, wires=Q_P1,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX), shots=SHOTS
    )
else:
    dev_p1 = qml.device("default.qubit", wires=Q_P1)

# ── Part II MRU device (4 qubits) ─────────────────────────────────────────────
Q_MRU = 4
if USE_BRAKET:
    dev_mru_hw = qml.device(
        "braket.aws.qubit", device_arn=QPU_ARN, wires=Q_MRU,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX), shots=SHOTS
    )
else:
    dev_mru_hw = None  # will be set after JAX check below

print(f"Backend : {'AWS Braket QPU' if USE_BRAKET else 'Local simulator'}")
print(f"Outputs : {OUTPUT_DIR.resolve()}")

# ── Data generation (same DGP as QFA_Solution2) ────────────────────────────────
def generate_regime_data(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    regime = rng.choice([1, 2], size=n, p=[0.75, 0.25])
    X1 = rng.normal(0, 1, n)
    X2 = cauchy.rvs(loc=0, scale=0.5, size=n, random_state=int(seed))
    rho = 0.6
    X3 = rho * X1 + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    X4 = rng.normal(0, 1, n)
    X  = np.column_stack([X1, X2, X3, X4])
    eps = rng.normal(0, 0.5, n)
    Y = np.where(regime == 1,
                 2*X1 - X2 + eps,
                 X1*X3 + np.log(np.abs(X2) + 1) + eps)
    return X, Y, regime

X_all, Y_all, R_all = generate_regime_data(n=20000, seed=42)
X_tr_raw, X_te_raw = X_all[:10000], X_all[10000:]
Y_tr, Y_te         = Y_all[:10000], Y_all[10000:]
R_tr, R_te         = R_all[:10000], R_all[10000:]

# ── Quantum preprocessing: winsorize → StandardScale → π·tanh(x/2) ────────────
def preprocess_quantum(X_tr, X_te):
    """Produces quantum-bounded inputs in (-π, π). No data leakage: fit on train only."""
    lo, hi = np.percentile(X_tr, [1, 99], axis=0)
    X_tr_c = np.clip(X_tr, lo, hi)
    X_te_c = np.clip(X_te, lo, hi)
    mu, sig = X_tr_c.mean(0), X_tr_c.std(0) + 1e-8
    X_tr_s  = np.clip((X_tr_c - mu) / sig, -5, 5)
    X_te_s  = np.clip((X_te_c - mu) / sig, -5, 5)
    return np.pi * np.tanh(X_tr_s / 2), np.pi * np.tanh(X_te_s / 2)

X_tr_q, X_te_q = preprocess_quantum(X_tr_raw, X_te_raw)
print(f"Part I  train : {X_tr_q.shape}  range [{X_tr_q.min():.3f}, {X_tr_q.max():.3f}]")
print(f"Part I  test  : {X_te_q.shape}")

# ── Angle+ZZ Circuit ─────────────────────────────────────────────────────────
@qml.qnode(dev_p1)
def angle_circuit(x):
    # Step 1: Angle embedding — Ry then Rz per qubit, no entanglement
    for i in range(4):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)
    # Step 2: ZZ feature map in series (hardware scaling: 2x)
    for _ in range(2):
        for i in range(4):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)
        for i in range(4):
            j = (i + 1) % 4
            qml.CNOT(wires=[i, j])
            qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
            qml.CNOT(wires=[i, j])
    obs  = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
            for i in range(4) for j in range(i+1, 4)]
    return obs

# ── Circuit resource summary ──────────────────────────────────────────────────
_x0 = np.zeros(4)
sp = qml.specs(angle_circuit)(_x0)["resources"]
print(f"  Angle+ZZ  qubits={sp.num_wires}  depth={sp.depth:3d}  "
      f"gates={sp.num_gates:3d}  outputs=10")

def extract_features(circuit_fn, X, label=""):
    """Run circuit on every row of X. Returns (N, n_obs) array."""
    t0 = time.time()
    feats = np.array([circuit_fn(xi) for xi in X])
    elapsed = time.time() - t0
    print(f"  {label:28s}: shape={feats.shape}  "
          f"({elapsed:.1f}s, {elapsed/len(X)*1000:.1f} ms/sample)")
    return feats

def save_p1_features(Q, y, regime, split, circuit_name):
    cols = [f"{circuit_name}_f{i}" for i in range(Q.shape[1])]
    df = pd.DataFrame(Q, columns=cols)
    df["y"]      = y
    df["regime"] = regime
    df["split"]  = split
    fpath = OUTPUT_DIR / f"part1_{circuit_name}_{split}.csv"
    df.to_csv(fpath, index=False)
    print(f"    → saved {fpath}")

print("=== Part I: train set ===")
Q_tr_angle = extract_features(angle_circuit, X_tr_q, "Angle (train)")

print("\n=== Part I: test set ===")
Q_te_angle = extract_features(angle_circuit, X_te_q, "Angle (test)")

print("\n=== Saving Part I features ===")
save_p1_features(Q_tr_angle, Y_tr, R_tr, "train", "angle")
save_p1_features(Q_te_angle, Y_te, R_te, "test",  "angle")

# Quick IC check (Pearson corr of each feature with y)
print("\n=== Part I: per-feature IC (test set) ===")
ics = [np.corrcoef(Q_te_angle[:, i], Y_te)[0, 1] for i in range(Q_te_angle.shape[1])]
print(f"  Angle+ZZ  max_IC={max(ics, key=abs):.4f}  "
      f"mean_|IC|={np.mean(np.abs(ics)):.4f}")

USE_REAL_DATA = True

MARKET  = "SPY"
PROXIES = ["SMH", "XLK", "XLY", "XLF"]

BUCKET_A_WEIGHTS = {
    "NVDA": 7.58, "AVGO": 2.64, "MSFT": 4.87, "AAPL": 6.66,
    "AMZN": 3.65, "GOOGL": 5.52, "META": 2.25,
}
BUCKET_B_WEIGHTS = {
    "TSLA": 1.90, "BRK-B": 1.56, "JPM": 1.39,
}
ALL_TICKERS = [MARKET] + PROXIES + list(BUCKET_A_WEIGHTS) + list(BUCKET_B_WEIGHTS)
START, END  = "2020-01-01", "2025-12-31"

if USE_REAL_DATA:
    try:
        import yfinance as yf
        raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)
        data_source = "yfinance (real)"
        print(f"Downloaded {len(raw)} rows for {len(ALL_TICKERS)} tickers")
    except Exception as e:
        print(f"yfinance failed ({e}) — falling back to synthetic")
        USE_REAL_DATA = False

if not USE_REAL_DATA:
    N_DAYS = 1259
    rng2   = np.random.RandomState(42)
    dates  = pd.bdate_range("2020-01-02", periods=N_DAYS)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols   = pd.MultiIndex.from_product([fields, ALL_TICKERS], names=["Price", "Ticker"])
    raw    = pd.DataFrame(index=dates, columns=cols)
    for t in ALL_TICKERS:
        rets  = rng2.normal(0.0005, 0.02, N_DAYS)
        close = 100 * np.exp(np.cumsum(rets))
        raw["Close",  t] = close
        raw["Open",   t] = close * (1 + rng2.normal(0, 0.005, N_DAYS))
        raw["High",   t] = raw[["Open", "Close"]].xs(t, level=1, axis=1).max(axis=1) * 1.01
        raw["Low",    t] = raw[["Open", "Close"]].xs(t, level=1, axis=1).min(axis=1) * 0.99
        raw["Volume", t] = 5e7 * (1 + np.abs(rng2.normal(0, 0.5, N_DAYS)))
    data_source = "synthetic fallback"

print(f"Data source: {data_source}")

w_A = {k: v / sum(BUCKET_A_WEIGHTS.values()) for k, v in BUCKET_A_WEIGHTS.items()}
w_B = {k: v / sum(BUCKET_B_WEIGHTS.values()) for k, v in BUCKET_B_WEIGHTS.items()}

def compute_rsi(close_series, period=10):
    delta = close_series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def build_bucket(df_raw, weights):
    idx = df_raw.index
    bucket_ret = pd.Series(0.0, index=idx)
    bucket_dv  = pd.Series(0.0, index=idx)
    for tic, w in weights.items():
        bucket_ret += w * df_raw["Close"][tic].pct_change().fillna(0)
        bucket_dv  += w * (df_raw["Close"][tic] * df_raw["Volume"][tic])
    close = [100.0]
    for r in bucket_ret.iloc[1:]:
        close.append(close[-1] * (1 + r))
    df = pd.DataFrame(index=idx)
    df["Close"]        = close
    df["Dollar_Volume"] = bucket_dv
    return df

bA_df = build_bucket(raw, w_A)
bB_df = build_bucket(raw, w_B)
print(f"Bucket A: {bA_df.shape}  Bucket B: {bB_df.shape}")

def extract_16_features(bucket_df, proxies_df, name, proxy1, proxy2):
    f    = pd.DataFrame(index=bucket_df.index)
    bc   = bucket_df["Close"]
    bo   = bucket_df.get("Open", bc)   # fallback if Open absent
    bdv  = bucket_df["Dollar_Volume"]
    spy_c  = proxies_df["Close"]["SPY"]
    spy_o  = proxies_df["Open"]["SPY"]
    def ret(s, n): return s.pct_change(n)

    f[f"ret_5_{name}_minus_SPY"]   = ret(bc, 5)   - ret(spy_c, 5)
    f[f"ret_60_{name}_minus_SPY"]  = ret(bc, 60)  - ret(spy_c, 60)
    f[f"ret_120_{name}_minus_SPY"] = ret(bc, 120) - ret(spy_c, 120)

    illiq_tic = (bc.pct_change().abs() / (bc + 1e-9)).rolling(20).mean()
    spy_dv    = spy_c * proxies_df["Volume"]["SPY"]
    illiq_spy = (spy_c.pct_change().abs() / (spy_dv + 1e-9)).rolling(20).mean()
    f[f"illiq_20_{name}_over_SPY"]  = illiq_tic / illiq_spy

    gap_b = (bo - bc.shift(1)) / bo
    gap_s = (spy_o - spy_c.shift(1)) / spy_o
    b_z   = (gap_b - gap_b.rolling(20).mean()) / (gap_b.rolling(20).std() + 1e-9)
    s_z   = (gap_s - gap_s.rolling(20).mean()) / (gap_s.rolling(20).std() + 1e-9)
    f[f"gap_zscore_diff_{name}_vs_SPY"] = b_z - s_z

    f[f"rsi10_{name}_minus_SPY"]  = compute_rsi(bc) - compute_rsi(spy_c)
    f[f"trend10_50_{name}_minus_SPY"] = (
        (bc.rolling(10).mean() / bc.rolling(50).mean()) - 1
      - (spy_c.rolling(10).mean() / spy_c.rolling(50).mean()) + 1
    )
    f[f"highpos20_{name}_minus_SPY"] = (
        (bc.rolling(20).max() / bc - 1)
      - (spy_c.rolling(20).max() / spy_c - 1)
    )
    dv_ma5 = bdv.rolling(5).mean()
    b_dvz  = (dv_ma5 - dv_ma5.rolling(60).mean()) / (dv_ma5.rolling(60).std() + 1e-9)
    s_dv_ma5 = spy_dv.rolling(5).mean()
    s_dvz  = (s_dv_ma5 - s_dv_ma5.rolling(60).mean()) / (s_dv_ma5.rolling(60).std() + 1e-9)
    f[f"dv_zscore_5_{name}_minus_SPY"] = b_dvz - s_dvz

    p1_c = proxies_df["Close"][proxy1]
    p2_c = proxies_df["Close"][proxy2]
    f[f"ret_20_{proxy1}_minus_SPY"] = ret(p1_c, 20) - ret(spy_c, 20)
    f[f"ret_60_{proxy1}_minus_SPY"] = ret(p1_c, 60) - ret(spy_c, 60)
    f[f"ret_20_{proxy2}_minus_SPY"] = ret(p2_c, 20) - ret(spy_c, 20)
    f[f"ret_60_{proxy2}_minus_SPY"] = ret(p2_c, 60) - ret(spy_c, 60)

    b_vol20 = bc.pct_change().rolling(20).std()
    s_vol20 = spy_c.pct_change().rolling(20).std()
    f[f"vol20_{name}_minus_SPY"] = b_vol20 - s_vol20

    def vw_return(close_s, vol_s, n=5):
        dr = close_s.pct_change()
        return (dr * vol_s).rolling(n).sum() / (vol_s.rolling(n).sum() + 1e-9)
    f[f"vwret_5_{name}_minus_SPY"] = vw_return(bc, bdv) - vw_return(spy_c, spy_dv)

    b_dvtrend = (bdv.rolling(10).mean() / (bdv.rolling(50).mean() + 1e-9)) - 1
    s_dvtrend = (spy_dv.rolling(10).mean() / (spy_dv.rolling(50).mean() + 1e-9)) - 1
    f[f"dvtrend10_50_{name}_minus_SPY"] = b_dvtrend - s_dvtrend

    f["target"] = (bc.shift(-5) / bc - 1) - (spy_c.shift(-5) / spy_c - 1)
    return f.dropna()

feat_A = extract_16_features(bA_df, raw, "BA", "SMH", "XLK")
feat_B = extract_16_features(bB_df, raw, "BB", "XLY", "XLF")
print(f"Bucket A features: {feat_A.shape}")
print(f"Bucket B features: {feat_B.shape}")

# Drop ret_120 duplicate introduced by Q4
feat_cols_A = [c for c in feat_A.columns if c != "target"]
feat_cols_B = [c for c in feat_B.columns if c != "target"]
feat_cols_A = list(dict.fromkeys(feat_cols_A))  # dedup preserving order
feat_cols_B = list(dict.fromkeys(feat_cols_B))
print(f"Unique features per bucket: A={len(feat_cols_A)}, B={len(feat_cols_B)}")

N_FEAT, R_MRU, D_MRU, M_MRU = 15, 4, 2, 10
# |W| = D * Q * R * 4 = 2*4*4*4 = 128 params

# ── JAX backend for vectorized batch evaluation ───────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _JAX_OK = True
    print("JAX available — using jit+vmap batch evaluation")
except ImportError:
    _JAX_OK = False
    print("JAX not found — using serial NumPy evaluation")

if USE_BRAKET:
    _dev_mru   = dev_mru_hw
    _INTERFACE = "numpy"
elif _JAX_OK:
    _dev_mru   = qml.device("default.qubit", wires=Q_MRU)
    _INTERFACE = "jax"
else:
    _dev_mru   = qml.device("default.qubit", wires=Q_MRU)
    _INTERFACE = "numpy"

@qml.qnode(_dev_mru, interface=_INTERFACE)
def mru_circuit(x, W):
    """
    Multiplexed Re-Uploading circuit.
    x: (N_FEAT,)  inputs in (-π, π)
    W: (D_MRU, Q_MRU, R_MRU, 4)  trainable params [w0, w1, b, w2] per slot
    Returns 15 Pauli-Z correlators (4 single + 6 two-body + 4 three-body + 1 four-body).
    """
    for d in range(D_MRU):
        for q in range(Q_MRU):
            for k in range(R_MRU):
                feat_idx = q * R_MRU + k
                w0, w1, b, w2 = W[d, q, k]
                qml.RZ(w0, wires=q)
                if feat_idx < N_FEAT:
                    qml.RY(w1 * x[feat_idx] + b, wires=q)
                else:
                    qml.RY(b, wires=q)              # bias pad
                qml.RZ(w2, wires=q)
        if d == 0:   # E1: nearest-neighbor ring
            qml.CZ(wires=[0, 1]); qml.CZ(wires=[1, 2])
            qml.CZ(wires=[2, 3]); qml.CZ(wires=[3, 0])
        else:        # E2: cross-diagonal
            qml.CZ(wires=[0, 2]); qml.CZ(wires=[1, 3])
    single = [qml.expval(qml.PauliZ(i)) for i in range(Q_MRU)]
    two    = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
              for i in range(Q_MRU) for j in range(i+1, Q_MRU)]
    three  = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j) @ qml.PauliZ(k))
              for i in range(Q_MRU)
              for j in range(i+1, Q_MRU)
              for k in range(j+1, Q_MRU)]
    four   = [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))]
    return single + two + three + four   # 4+6+4+1 = 15

# ── Vectorized batch evaluator ─────────────────────────────────────────────────
if _JAX_OK and not USE_BRAKET:
    _mru_batch_jax = jax.jit(jax.vmap(mru_circuit, in_axes=(0, None)))
    def _mru_batch(X, W):
        out = _mru_batch_jax(jnp.array(X), jnp.array(W))
        # PennyLane may return a tuple of (batch,) arrays — one per observable
        if isinstance(out, (list, tuple)):
            out = jnp.stack(out, axis=1)  # → (batch, n_obs)
        return np.array(out)
else:
    def _mru_batch(X, W):
        return np.array([mru_circuit(xi, W) for xi in X])

def init_W(seed=None):
    """Near-identity init: w1=1, w0/w2~N(0,0.01)."""
    rng = np.random.RandomState(seed)
    W = np.zeros((D_MRU, Q_MRU, R_MRU, 4))
    W[:, :, :, 1] = 1.0
    W[:, :, :, 0] = rng.randn(D_MRU, Q_MRU, R_MRU) * 0.01
    W[:, :, :, 3] = rng.randn(D_MRU, Q_MRU, R_MRU) * 0.01
    return W

def extract_mru_features(X, W, obs_idx):
    """Extract M_MRU features for all rows of X."""
    return _mru_batch(X, W)[:, obs_idx]

def spsa_train(W_init, X_tr, y_tr, n_steps=50, n_warmup=10,
               batch=16, a=0.05, c=0.1, A=10, seed=0):
    """
    SPSA training for MRU. 2 circuit evals/step.
    Loss: MSE of lstsq fit on quantum features (no Ridge regularization).
    Returns (W_opt, obs_idx).
    """
    rng     = np.random.RandomState(seed)
    W_flat  = W_init.ravel().copy()
    n_p     = len(W_flat)
    obs_idx = np.arange(15)

    def loss_fn(W_f, o_idx):
        F = _mru_batch(X_batch, W_f.reshape(W_init.shape))[:, o_idx]
        # Simple lstsq (no regularization — pure quantum feature quality)
        coef, _, _, _ = np.linalg.lstsq(F, y_batch, rcond=None)
        resid = y_batch - F @ coef
        return float(resid @ resid) / len(y_batch)

    for step in range(n_steps):
        if step == n_warmup:
            W_shaped = W_flat.reshape(W_init.shape)
            n_var    = min(32, len(X_tr))
            idx_v    = rng.choice(len(X_tr), size=n_var, replace=False)
            F_all    = _mru_batch(X_tr[idx_v], W_shaped)
            obs_idx  = np.argsort(F_all.var(axis=0))[::-1][:M_MRU]

        idx_b   = rng.choice(len(X_tr), size=min(batch, len(X_tr)), replace=False)
        X_batch = X_tr[idx_b]
        y_batch = y_tr[idx_b]

        ak    = a / (step + 1 + A) ** 0.602
        ck    = c / (step + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=n_p)
        L_p   = loss_fn(W_flat + ck * delta, obs_idx)
        L_m   = loss_fn(W_flat - ck * delta, obs_idx)
        W_flat -= ak * (L_p - L_m) / (2 * ck) * delta

    if n_steps <= n_warmup:
        W_shaped = W_flat.reshape(W_init.shape)
        n_var    = min(32, len(X_tr))
        idx_v    = rng.choice(len(X_tr), size=n_var, replace=False)
        F_all    = _mru_batch(X_tr[idx_v], W_shaped)
        obs_idx  = np.argsort(F_all.var(axis=0))[::-1][:M_MRU]

    return W_flat.reshape(W_init.shape), obs_idx

# ── Smoke test ─────────────────────────────────────────────────────────────────
_W0  = init_W(seed=0)
_out = _mru_batch(np.zeros((1, N_FEAT)), _W0)

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

_sp = qml.specs(_mru_for_specs)(np.zeros(N_FEAT), _W0)["resources"]
print(f"MRU  qubits={_sp.num_wires}  depth={_sp.depth}  "
      f"gates={_sp.num_gates}  params={_W0.size}  all_obs=15  selected={M_MRU}")
print(f"Batch: {'JAX jit+vmap' if (_JAX_OK and not USE_BRAKET) else 'NumPy serial'}")

# ── Walk-forward knobs ─────────────────────────────────────────────────────────
DEBUG_MODE       = False    # True = first 20 windows only
SPSA_STEPS       = 250
SPSA_WARMUP      = 32
SPSA_BATCH       = 16
SPSA_A           = 0.05
SPSA_C           = 0.1
SPSA_A_OFFSET    = 10
RETRAIN_EVERY_K  = 5        # retrain MRU every K windows, warm-start otherwise
LOOKBACK_WINDOW  = 104      # trading days of training history

BUCKETS = [
    (feat_A, feat_cols_A, "Bucket_A"),
    (feat_B, feat_cols_B, "Bucket_B"),
]

all_feat_records  = []   # quantum features per window
all_pred_records  = []   # quantum predictions per window

for feat_df, feat_cols, bucket_name in BUCKETS:
    print(f"\n{'='*60}")
    print(f"Bucket: {bucket_name}  ({len(feat_df)} rows, {len(feat_cols)} features)")

    X_all_raw = feat_df[feat_cols].values.astype(np.float64)
    y_all     = feat_df["target"].values.astype(np.float64)
    dates_all = feat_df.index

    n_total   = len(X_all_raw)
    W_prev    = None
    obs_prev  = np.arange(M_MRU)
    window_count = 0

    for i in range(LOOKBACK_WINDOW + 5, n_total):
        if DEBUG_MODE and window_count >= 20:
            break

        # ── Train/test split ──────────────────────────────────────────────────
        X_tr_raw_w = X_all_raw[i - LOOKBACK_WINDOW - 5 : i - 5]
        y_tr_w     = y_all[i - LOOKBACK_WINDOW - 5 : i - 5]
        X_te_raw_w = X_all_raw[i : i + 1]
        y_te_w     = y_all[i : i + 1]
        test_date  = dates_all[i]

        # ── Quantum preprocessing ─────────────────────────────────────────────
        lo, hi  = np.percentile(X_tr_raw_w, [1, 99], axis=0)
        X_tr_c  = np.clip(X_tr_raw_w, lo, hi)
        X_te_c  = np.clip(X_te_raw_w, lo, hi)
        mu, sig = X_tr_c.mean(0), X_tr_c.std(0) + 1e-8
        X_tr_s  = np.clip((X_tr_c - mu) / sig, -5, 5)
        X_te_s  = np.clip((X_te_c - mu) / sig, -5, 5)
        X_tr_q_w = np.pi * np.tanh(X_tr_s / 2)   # (LOOKBACK, N_FEAT)
        X_te_q_w = np.pi * np.tanh(X_te_s / 2)   # (1, N_FEAT)

        # ── MRU training ──────────────────────────────────────────────────────
        retrain = (window_count % RETRAIN_EVERY_K == 0) or (W_prev is None)
        if retrain:
            init_W_val = init_W(seed=window_count) if W_prev is None else W_prev.copy()
            W_opt, obs_idx = spsa_train(
                init_W_val, X_tr_q_w, y_tr_w,
                n_steps=SPSA_STEPS, n_warmup=SPSA_WARMUP,
                batch=SPSA_BATCH, a=SPSA_A, c=SPSA_C, A=SPSA_A_OFFSET,
                seed=window_count,
            )
            W_prev   = W_opt
            obs_prev = obs_idx
        else:
            W_opt, obs_idx = W_prev, obs_prev

        # ── Extract quantum features ───────────────────────────────────────────
        Q_tr_mru = extract_mru_features(X_tr_q_w, W_opt, obs_idx)  # (LOOKBACK, M_MRU)
        Q_te_mru = extract_mru_features(X_te_q_w, W_opt, obs_idx)  # (1, M_MRU)

        # ── Quantum-only prediction (lstsq, no Ridge) ─────────────────────────
        coef, _, _, _ = np.linalg.lstsq(Q_tr_mru, y_tr_w, rcond=None)
        y_hat_mru = float(Q_te_mru @ coef)

        # ── Record quantum features for this window ────────────────────────────
        feat_row = {"bucket": bucket_name, "date": test_date}
        for m in range(M_MRU):
            feat_row[f"mru_f{m}"] = float(Q_te_mru[0, m])
        feat_row["actual"] = float(y_te_w[0])
        all_feat_records.append(feat_row)

        pred_row = {
            "bucket":     bucket_name,
            "date":       test_date,
            "y_hat_mru":  y_hat_mru,
            "actual":     float(y_te_w[0]),
            "retrained":  int(retrain),
        }
        all_pred_records.append(pred_row)

        window_count += 1
        if window_count % 20 == 0:
            print(f"  window {window_count:4d}/{n_total - LOOKBACK_WINDOW - 5}  "
                  f"date={test_date.date()}  retrain={retrain}")

    # Save trained weights for this bucket
    if W_prev is not None:
        w_path = OUTPUT_DIR / f"mru_weights_{bucket_name}.npy"
        np.save(w_path, W_prev)
        print(f"  Weights saved → {w_path}")

print("\nWalk-forward complete.")

# ── Save Part II results to CSV ───────────────────────────────────────────────
feat_df_out = pd.DataFrame(all_feat_records)
pred_df_out = pd.DataFrame(all_pred_records)

feat_path = OUTPUT_DIR / "part2_mru_features.csv"
pred_path = OUTPUT_DIR / "part2_mru_predictions.csv"
feat_df_out.to_csv(feat_path, index=False)
pred_df_out.to_csv(pred_path, index=False)
print(f"Saved: {feat_path}  ({len(feat_df_out)} rows)")
print(f"Saved: {pred_path}  ({len(pred_df_out)} rows)")

# ── Per-bucket IC summary ─────────────────────────────────────────────────────
print("\n=== Part II: Quantum MRU IC Summary ===")
print(f"{'Bucket':12s}  {'IC (Pearson)':>14s}  {'IC (Spearman)':>14s}  {'N':>6s}")
print("-" * 52)
for bname, grp in pred_df_out.groupby("bucket"):
    ic_p = float(np.corrcoef(grp["y_hat_mru"], grp["actual"])[0, 1])
    from scipy.stats import spearmanr
    ic_s, _ = spearmanr(grp["y_hat_mru"], grp["actual"])
    print(f"{bname:12s}  {ic_p:>14.4f}  {ic_s:>14.4f}  {len(grp):>6d}")

# ── Part I IC summary ─────────────────────────────────────────────────────────
print("\n=== Part I: Per-circuit max |IC| (test) ===")
ics = [np.corrcoef(Q_te_angle[:, i], Y_te)[0, 1] for i in range(Q_te_angle.shape[1])]
best_ic = max(ics, key=abs)
print(f"  Angle+ZZ  best IC={best_ic:+.4f}  "
      f"R1 best={max([np.corrcoef(Q_te_angle[R_te==1, i], Y_te[R_te==1])[0,1] for i in range(Q_te_angle.shape[1])], key=abs):+.4f}  "
      f"R2 best={max([np.corrcoef(Q_te_angle[R_te==2, i], Y_te[R_te==2])[0,1] for i in range(Q_te_angle.shape[1])], key=abs):+.4f}")

print(f"\nAll quantum outputs saved to: {OUTPUT_DIR.resolve()}")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

# ── Part II: IC Summary Bar + Rolling IC ─────────────────────────────────────
buckets   = pred_df_out['bucket'].unique()
ic_pearson  = []
ic_spearman = []
for b in buckets:
    grp = pred_df_out[pred_df_out['bucket'] == b]
    ic_pearson.append(float(np.corrcoef(grp['y_hat_mru'], grp['actual'])[0, 1]))
    ic_s, _ = spearmanr(grp['y_hat_mru'], grp['actual'])
    ic_spearman.append(float(ic_s))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: per-bucket IC bars
x = np.arange(len(buckets))
w = 0.35
axes[0].bar(x - w/2, ic_pearson,  w, label='Pearson IC',  color='#E24A33', alpha=0.85)
axes[0].bar(x + w/2, ic_spearman, w, label='Spearman IC', color='#4878CF', alpha=0.85)
axes[0].axhline(0, c='k', lw=0.7)
axes[0].set_xticks(x)
axes[0].set_xticklabels(buckets, rotation=45)
axes[0].set_ylabel('IC')
axes[0].set_title('Part II — MRU Quantum IC by Bucket')
axes[0].legend(fontsize=9)

# Right: rolling IC over time per bucket
ROLL = 30
pred_df_out['date'] = pd.to_datetime(pred_df_out['date'])
colors = ['#E24A33', '#4878CF', '#6ACC65', '#8B4513']
for idx_b, b in enumerate(buckets):
    grp = pred_df_out[pred_df_out['bucket'] == b].sort_values('date')
    rolling_ic = [
        float(np.corrcoef(grp['y_hat_mru'].iloc[max(0, i-ROLL):i],
                          grp['actual'].iloc[max(0, i-ROLL):i])[0, 1])
        if i >= 5 else float('nan')
        for i in range(1, len(grp)+1)
    ]
    axes[1].plot(grp['date'].values, rolling_ic,
                 label=b, color=colors[idx_b % len(colors)], lw=1.2)
axes[1].axhline(0, c='k', lw=0.7, ls='--')
axes[1].set_ylabel(f'{ROLL}-day Rolling IC')
axes[1].set_title('Part II — MRU Rolling IC Over Time')
axes[1].legend(fontsize=9)
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('part2_quantum_ic.png', bbox_inches='tight')
plt.show()
print('Saved: part2_quantum_ic.png')

# ── Part II: Predicted vs Actual Scatter per Bucket ───────────────────────────
n_buckets = len(buckets)
fig, axes = plt.subplots(1, n_buckets, figsize=(6 * n_buckets, 5))
if n_buckets == 1:
    axes = [axes]

for ax, b in zip(axes, buckets):
    grp = pred_df_out[pred_df_out['bucket'] == b]
    ic  = float(np.corrcoef(grp['y_hat_mru'], grp['actual'])[0, 1])
    ax.scatter(grp['y_hat_mru'], grp['actual'], alpha=0.3, s=8, color='#E24A33')
    # Regression line
    m, b_ = np.polyfit(grp['y_hat_mru'], grp['actual'], 1)
    xl = np.linspace(grp['y_hat_mru'].min(), grp['y_hat_mru'].max(), 100)
    ax.plot(xl, m * xl + b_, 'k--', lw=1.2)
    ax.set_xlabel('MRU Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{b}\nIC={ic:+.4f}')

plt.tight_layout()
plt.savefig('part2_quantum_scatter.png', bbox_inches='tight')
plt.show()
print('Saved: part2_quantum_scatter.png')

# ── Part I: Angle+ZZ Feature IC Heatmap ──────────────────────────────────────
n_obs = Q_te_angle.shape[1]
feat_labels = ([f'Z{i}' for i in range(4)] +
               [f'Z{i}Z{j}' for i in range(4) for j in range(i+1, 4)])

ic_all = np.array([np.corrcoef(Q_te_angle[:, i], Y_te)[0, 1] for i in range(n_obs)])
ic_r1  = np.array([np.corrcoef(Q_te_angle[R_te==1, i], Y_te[R_te==1])[0, 1] for i in range(n_obs)])
ic_r2  = np.array([np.corrcoef(Q_te_angle[R_te==2, i], Y_te[R_te==2])[0, 1] for i in range(n_obs)])

heatmap_data = np.vstack([ic_all, ic_r1, ic_r2])  # (3, n_obs)

fig, ax = plt.subplots(figsize=(14, 3))
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['All', 'Regime 1', 'Regime 2'])
ax.set_xticks(range(n_obs))
ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=9)
ax.set_title('Part I — Angle+ZZ Feature IC with Target (test set)')
plt.colorbar(im, ax=ax, label='Pearson IC')
plt.tight_layout()
plt.savefig('part1_quantum_features.png', bbox_inches='tight')
plt.show()
print('Saved: part1_quantum_features.png')

print("=" * 65)
print("QUANTUM RESOURCE REPORT")
print("=" * 65)

print("\nPart I — Feature Map Circuits (4 qubits each)")
print(f"  {'Circuit':8s}  {'Qubits':>6s}  {'Depth':>6s}  {'Gates':>6s}  {'Outputs':>8s}")
print("  " + "-" * 42)
_x0 = np.zeros(4)
for cname, cfn, n_out in [("Angle+ZZ", angle_circuit, 10)]:
    sp = qml.specs(cfn)(_x0)["resources"]
    print(f"  {cname:8s}  {sp.num_wires:>6d}  {sp.depth:>6d}  {sp.num_gates:>6d}  {n_out:>8d}")

print("\nPart II — MRU Variational Circuit")
sp = qml.specs(_mru_for_specs)(np.zeros(N_FEAT), _W0)["resources"]
print(f"  Qubits         : {sp.num_wires}")
print(f"  Depth          : {sp.depth}")
print(f"  Gates          : {sp.num_gates}")
print(f"  Trainable params: {_W0.size}")
print(f"  Observables    : 15 (top {M_MRU} selected by variance)")
print(f"  SPSA steps     : {SPSA_STEPS}  (2 circuit evals/step)")
print(f"  Circuit evals/window: {2 * SPSA_STEPS} (train) + {LOOKBACK_WINDOW + 1} (extract)")

total_windows = sum(len(grp) for _, grp in pred_df_out.groupby("bucket"))
n_retrain = pred_df_out["retrained"].sum()
print(f"\n  Total windows  : {total_windows}")
print(f"  MRU retrains   : {n_retrain}  (every {RETRAIN_EVERY_K} windows)")
print(f"  Total circuit evals (MRU): ~{n_retrain * 2 * SPSA_STEPS + total_windows * (LOOKBACK_WINDOW + 1):,}")

print("\n" + "=" * 65)
print("Notebook complete. Outputs in quantum_outputs/")
print("=" * 65)
