# YQuantum_2026_AWS_x_State_Street_Team_5

# YQuantum 2026 - AWS x State Street Challenge

## Quantum Feature Augmentation for Financial Market Prediction

## 1. Motivation and Core Difficulty

Financial market prediction is characterized by:

- **Low signal-to-noise ratios**
- **Non stationarity and regime shifts**
- **Heavy tails and non-Gaussian distributions**
- **Strong risk of overfitting**

In such settings, **feature engineering** often matters more than the choice of model. The core hypothesis behind this challenge is:

*Quantum derived feature transformations may expose nonlinear structure in the data that classical feature maps fail to capture, without materially increasing overfitting.*

The challenge is **not** to demonstrate quantum supremacy, but to rigorously test whether **quantum feature augmentation provides incremental, measurable value** relative to strong classical baselines under realistic out-of-sample evaluation.

## 2. Core Scientific Question

Formally:

**Do quantum derived feature transformations improve out-of-sample predictive performance for financial prediction tasks, relative to classical feature engineering, when evaluated under strict train/test separation and rolling backtests?**

Both outcomes are valid:

- *Positive result*: Quantum features improve generalization
- *Negative result*: No improvement, or worse generalization

What matters most is **experimental rigor, transparency, and reproducibility.**

## 3. Prediction Tasks Overview

Participants must evaluate quantum feature augmentation on **two tasks**:

1. **Synthetic Regime Switching Process**
2. **Real Financial Data: S&P 500 Stock Excess Returns**

The synthetic task ensures controlled, interpretable results, while the real-data task introduces non-stationarity, market structure, and practical constraints.

---

## Part I — Simulated Regime Switching Process

## 4. Example Data Generating Process (DGP) (feel free to make the DGP more or less complicated depending on results)

The target variable $Y$ is generated from a **latent regime switching model**:

$$Y = \begin{cases} 2X_{1} - X_{2} + \varepsilon, & \text{if Regime 1} \\ (X_{1} \cdot X_{3}) + \log(|X_{2}| + 1) + \varepsilon, & \text{if Regime 2} \end{cases}$$

**Regime Probabilities**

$$P(\text{Regime 1}) = 0.75, \quad P(\text{Regime 2}) = 0.25$$

The regime indicator is **latent** unless explicitly provided as an experimental variant.

## 5. Feature Distributions

**Regime 1**

$$\begin{array}{ll} X_{1}, X_{2} & \sim \mathcal{N}(0,1) \\ X_{3} & \sim \mathcal{N}(0,1) \\ X_{4} & \sim \text{Uniform}(-1,1) \\ \varepsilon & \sim \mathcal{N}(0,1) \end{array}$$

**Regime 2**

$$\begin{array}{ll} (X_{1}, X_{3}) & \sim \mathcal{N}\!\left(\begin{bmatrix} 3 \\ 3 \end{bmatrix}, \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}\right) \\ X_{2} & \sim \text{Cauchy}(0,1) \\ X_{4} & \sim \text{Exponential}(\lambda = 1) \\ \varepsilon & \sim \mathcal{N}(0,1) \end{array}$$

**Key Design Intentions**

| **Variable** | **Purpose** |
|---|---|
| $X_{1}$ | Linear signal in Regime 1 |
| $X_{2}$ | Linear (R1), nonlinear heavy‑tailed (R2) |
| $X_{3}$ | Only relevant in Regime 2 |
| $X_{4}$ | Pure noise |
| Regime | Hidden non stationarity |

This setup deliberately creates:

- **Regime dependent relevance**
- **Nonlinear interactions**
- **Distributional shifts**
- **Spurious features**

## 6. Baseline Features

The model observes only:

$$\mathbf{X} = (X_{1}, X_{2}, X_{3}, X_{4})$$

No regime indicator is provided by default.

## 7. Feature Augmentation Strategies

### 7.1 Classical Feature Augmentation (Baseline)

Let $\phi_{c}(\mathbf{X})$ denote classical transformations such as:

- Polynomial terms:

$$\{ X_{i},\ X_{i}^{2},\ X_{i}^{3} \}$$

- Interaction terms:

$$\{ X_{i} X_{j} \mid i \neq j \}$$

- Log or absolute transforms:

$$\log x,|x|$$

The classical augmented feature vector is:

$$\widetilde{\mathbf{X}}^{(c)} = \phi_{c}(\mathbf{X})$$

Include the original features $${X}$$ in $$\widetilde{\mathbf{X}}^{(c)}$$

### 7.2 Quantum Feature Generation

Quantum features are generated via a **parameterized quantum feature map**:

$$\phi_{q}(\mathbf{X}) = \langle \psi(\mathbf{X}) | \widehat{O} | \psi(\mathbf{X}) \rangle$$

Where:

- $|\psi(\mathbf{X})\rangle = U(\mathbf{X}) |0\rangle^{\otimes n}$
- $U(\mathbf{X})$ is a unitary parameterized by classical input
- $\widehat{O}$ is a measurement operator (e.g., single-qubit Pauli $Z_k$, pairwise $Z_k Z_l$, or higher-weight Pauli strings)

There are many ways to encode classical input into quantum states, with angle encoding being a typical starting point. A simple encoding circuit applies a layer of single-qubit rotations followed by entangling gates:

$$U(\mathbf{X}) = W_{\text{ent}} \prod_{k=1}^{n} R_Z^{(k)}(X_k)$$

where $R_Z^{(k)}(X_k)$ applies a $Z$-rotation with angle $X_k$ on the $k$-th qubit and $W_{\text{ent}}$ is a fixed entangling layer (e.g., a ladder of CNOTs).

This produces **nonlinear, high-dimensional embeddings** inaccessible to linear classical transforms.

The augmented feature vector for a single feature is:

$$\widetilde{\mathbf{X}}^{(q)} = [\mathbf{X},\ \phi_{q}(\mathbf{X})]$$

The focus is on **feature generation**, not increasing model complexity.

Include the original features $${X}$$ in $$\widetilde{\mathbf{X}}^{(q)}$$

### 7.3 Dimensionality Control

To avoid overfitting:

- Ridge regression:

$$\min_{\beta} \| Y - X\beta \|^{2} + \lambda \| \beta \|^{2}$$

- Lasso:

$$\min_{\beta} \| Y - X\beta \|^{2} + \lambda \| \beta \|_{1}$$

The same regularization strategy must be applied consistently across classical and quantum features.

## 8. Model Requirement

At minimum, participants must include **linear regression**:

$$\widehat{Y} = \beta_{0} + \sum_{j} \beta_{j} \widetilde{X}_{j}$$

**Important constraint**:

The **same model class** must be used for classical and quantum features.

This isolates the effect of **feature transformations**, not model complexity.

Optional extensions (e.g., kernels, trees) may be explored, but results should clearly distinguish feature effects from modeling effects.

## 9. Experimental Setup (Synthetic Task)

- Training set: 10,000 observations
- Test set: 10,000 observations
- Optional validation set for tuning
- Multiple random seeds encouraged

**Metrics**

- Mean Squared Error (MSE):

$$\text{MSE} = \frac{1}{N} \sum (Y - \widehat{Y})^{2}$$

- Mean Absolute Error (MAE)

- Correlation:

$$\rho(Y, \widehat{Y})$$

---

## Part II — Predicting Stock Excess Returns

The main goal of this exercise is not to find the best set of features and best models to predict the market, but to evaluate if there is additional benefit from the quantum features compared to the classical approach.

## 10. Target Variable

Predict the next 5-day excess returns of S&P 500 constituents, where excess return is defined as

For stock $i$:

$$Y_{i,t} = R_{i,t}^{(5d)} - R_{\mathrm{SP500},t}^{(5d)}$$

This removes market wide effects and focuses on **relative performance**. You can start with the current largest 10 stocks or so and expand if time permits.

## 11. Feature Construction

All features are defined as **stock minus market** values. We suggest using data available on yfinanceAPI, focusing primarily on price (high, low, open, close) and volume data. Creative variables constructed based on price and volume are welcome, as long as they are logically constructed and clearly documented.

**Examples**

Price related

- prior 5-day return of stock subtracted by that of market (stock – market)
- prior 20-day return (stock – market)
- prior 120-day return (stock – market)
- max price (the max daily "High" over prior 20 days) / current price – 1 (stock – market)
- min price (the min daily "Low" over prior 20 days) / current price – 1 (stock – market)
- Relative Strength Index (RSI) over prior 10 days ([Relative Strength Index (RSI): What It Is, How It Works, and Formula](https://www.investopedia.com/terms/r/rsi.asp)) (stock – market)
- Price trend: $\dfrac{MA_{10}}{MA_{50}} - 1$ (stock – market)

Volume related

- Z-score of past 5-day average volume (stock – market)
- Z-score of past 20-day average volume (stock – market)
- Volume trend: $\dfrac{MA_{10}}{MA_{50}} - 1$ (stock – market)

Price and Volume

- Volume-weighted return $\displaystyle\sum_{d=1}^{5} w_{d} \cdot r_{d}$, where $w_{d} = \dfrac{V_{d}}{\sum_{k=1}^{5} V_{k}}$ (stock – market)
- Volume-weighted return – actual return (stock – market)
- Prior 5-day return × Z-score of 5-day volume (stock – market)
- RSI × Z-score of 5-day volume (stock – market)
- Volume trend – price trend (stock – market)

## 12. Augmentation and Modeling

Same framework as synthetic task:

- Classical polynomial/interactions
- Quantum feature maps
- Linear regression baseline
- Optional extensions (tree models, kernels)

## 13. Walk Forward Back test

Training is performed using **rolling windows**, not random splits.

At each time $t$:

1. Train on past 2 years
2. Validate (optional)
3. Predict next 5‑day return
4. Roll window forward by one day

This mimics **real trading conditions** and prevents look‑ahead bias.

## 14. Evaluation Expectations

Participants should report:

- Aggregate out-of-sample MSE / MAE
- Information coefficient (correlation)
- Performance stability across time
- Feature count vs performance tradeoff
- Overfitting diagnostics

## 15. What a Strong Submission Demonstrates

- Clean experimental design.
- Apples-to-apples comparisons.
- Honest reporting of negative results.
- Careful control of overfitting.
- Clear discussion of why quantum features help or fail.
- Financial realism in evaluation.
- Quantum Resource Usage Is a Required Result Dimension (memory, circuit depth, qubit count, etc).
- Cost–performance trade-offs is part of result interpretations.

## 16. AWS Workshop Studio Access Instructions:
You'll have access to AWS Workshop Studio for the duration of the challenge.

### Steps to access AWS Workshop Studio:

- Go to [AWS Workshop Studio](https://catalog.workshops.aws)
- Click on "Get Started"
- Select sign-in option "Email one-time password (OTP)"
- Enter your email address, click on "Send passcode" and check your mailbox for your OTP
- Enter the OTP and click on "Sign in"
- Provide the 12-digit access code provided by an AWS representative during the event: 2447-0a1960-dc
- Review and accept terms and conditions to join the event
- In the navigation pane on the left on the event landing page, click on "Open AWS console" to access the provided AWS account with Braket pre-configured

## 17. Braket Resources to get started:

- [Exact simulation of Quantum Enhanced Signature Kernels for financial data streams prediction using Amazon Braket](https://aws.amazon.com/blogs/quantum-computing/exact-simulation-of-quantum-enhanced-signature-kernels-for-financial-data-streams-prediction-using-amazon-braket/)
- [Quantum Kernels Predict Financial Data With Amazon Braket](https://quantumzeitgeist.com/amazon-braket-quantum-kernels-financial-prediction/)

**Local**

- [Testing a quantum task with the local simulator - Amazon Braket](https://docs.aws.amazon.com/braket/latest/developerguide/braket-send-to-local-simulator.html)
- [LocalSimulator — amazon-braket-sdk 1.113.1.dev0 documentation](https://amazon-braket-sdk-python.readthedocs.io/en/latest/_apidoc/braket.devices.local_simulator.html)

**Hybrid**

- [Working with Amazon Braket Hybrid Jobs - Amazon Braket](https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs.html)
- [Amazon Braket Hybrid Jobs — amazon-braket-sdk 1.113.1.dev0 documentation](https://amazon-braket-sdk-python.readthedocs.io/en/latest/examples-hybrid-jobs.html)

**Examples:**

- [GitHub - amazon-braket/amazon-braket-examples: Example notebooks that show how to apply quantum computing](https://github.com/amazon-braket/amazon-braket-examples)