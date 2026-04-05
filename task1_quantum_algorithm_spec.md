# Task 1 Quantum Feature Augmentation Algorithm Spec

Implement the notes-file algorithm exactly as a **fixed quantum feature augmentation block** inside the notebook.

Given one sample \(x=(x_1,x_2,x_3,x_4)\), construct a feature vector

\[
\zeta(x)=
\begin{bmatrix}
c(x)\\
f(x)
\end{bmatrix}
\]

where:
- \(c(x)\) = classical features you already decided to keep
- \(f(x)\) = quantum measurement features from the fixed circuit

Then fit only a classical linear model

\[
\hat y = b + w^\top \zeta(x)
\]

and train by updating only \(b,w\) from the regression loss. No quantum parameters are trained. That is exactly what page 2 of your notes now says.

## Exact algorithm to implement from the notes

For each input sample \(x=(x_1,x_2,x_3,x_4)\):

### 1. Build the single-feature transformed inputs

Use the per-feature map shown in the notes:

\[
\phi(x)=
\begin{bmatrix}
zscore(x_1)\\
\ln(1+|x_2|)\operatorname{sgn}(x_2)\\
x_3\\
x_4
\end{bmatrix}
\]

So define:
- \(\phi_1(x_1)=zscore(x_1)\)
- \(\phi_2(x_2)=\log(1+|x_2|)\,\mathrm{sign}(x_2)\)
- \(\phi_3(x_3)=x_3\)
- \(\phi_4(x_4)=x_4\)

Important implementation detail:
- compute the z-score normalization statistics from the training set only
- then apply those same stats to val/test
- for consistency, I would standardize all 4 raw inputs before feeding them anywhere, but if you want to follow the notes literally, only \(x_1\) gets explicit z-score inside \(\phi\)

### 2. Define pairwise interaction functions

Use the pairwise map from the notes:

\[
\phi_{ij}(x_i,x_j)=x_i x_j
\]

for whichever qubit pairs you include. Since you have 4 qubits/features, the natural full set is:

\[
(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
\]

This matches your “encode as pairwise interactions” construction.

### 3. Use 4 qubits, one per feature

Initialize the quantum state as

\[
|0000\rangle
\]

### 4. Apply the single-feature encoding block \(B(x)\)

Your notes define the single-feature nonlinearity as an \(R_y\) followed by an \(R_z\) on each qubit, using the transformed scalar \(\phi_i(x_i)\). Implement:

\[
B(x)=\prod_{i=1}^{4}\left(R_y^{(i)}(\phi_i(x_i))\,R_z^{(i)}(\phi_i(x_i))\right)
\]

In Pennylane terms, on wire `i`:

```python
qml.RY(phi_i, wires=i)
qml.RZ(phi_i, wires=i)
```

Apply this for all 4 wires. This is the “single feature nonlinearity: Ry() Rz()” block in the notes.

### 5. Apply the pairwise interaction block \(A(x)\)

Your notes define

\[
A(x)=\exp\left(i\left[\sum_i \phi_i(x_i) Z_i + \sum_{i<j}\phi_{ij}(x_i,x_j) Z_i Z_j\right]\right)
\]

In code, implement this as commuting Pauli phase evolutions:
- a phase term on each qubit from \(\phi_i(x_i)\)
- a pairwise \(ZZ\) phase term for each pair from \(x_ix_j\)

The clean implementation in Pennylane is:
- for each wire \(i\), apply another `qml.RZ(2 * phi_i, wires=i)` if you want the single-qubit \(Z_i\) phase contribution represented directly
- for each chosen pair \((i,j)\), apply a \(ZZ\)-interaction gate with angle proportional to \(x_i x_j\)

Depending on device / Pennylane version, coding agent should use one of:

```python
qml.IsingZZ(theta, wires=[i, j])
```

or an equivalent decomposition.

So for each pair:

\[
\theta_{ij} = x_i x_j
\]

and apply

```python
qml.IsingZZ(theta_ij, wires=[i, j])
```

This is the implementation of the notes’ pairwise interaction exponential. Since all terms are diagonal in \(Z\), this is the natural gate-level realization.

### 6. Full circuit

The full unitary from your notes is

\[
U(x)=B(x)A(x)
\]

and the encoded state is

\[
|\psi(x)\rangle = U(x)|0000\rangle
\]

So coding order should be:

1. initialize 4-wire device  
2. apply \(B(x)\): all single-qubit `RY`, `RZ`
3. apply \(A(x)\): single-qubit `RZ` phase terms and pairwise `IsingZZ`
4. measure observables

That is the exact architecture shown on page 1.

## Measurement features to return

Your notes explicitly show basis measurements:

\[
f(\psi(x))=
\begin{bmatrix}
\langle Z_1\rangle\\
\langle Z_2\rangle\\
\langle Z_3\rangle\\
\langle Z_1 Z_3\rangle
\end{bmatrix}
\]

So implement exactly that 4-dimensional quantum feature vector first, since that is what your page shows.

In Pennylane QNode return:

```python
return [
    qml.expval(qml.PauliZ(0)),
    qml.expval(qml.PauliZ(1)),
    qml.expval(qml.PauliZ(2)),
    qml.expval(qml.PauliZ(0) @ qml.PauliZ(2)),
]
```

Why \(\langle Z_1Z_3\rangle\): because your notes emphasize \(x_1,x_3\) as the meaningful pairwise interaction to encode. So the measured pair feature should match that. Do not bloat this unless needed later.

## Final augmented feature vector

For each sample, construct:

\[
\zeta(x)=
\begin{bmatrix}
c(x)\\
f(x)
\end{bmatrix}
\]

where:
- `c(x)` is your classical feature vector already chosen elsewhere
- `f(x)` is the 4-d quantum feature vector above

So in code:

```python
zeta = np.concatenate([c_x, f_x], axis=0)
```

Your notes literally label this as “classical features” stacked with “quantum feature augmentation.”

## Training step

Then fit:

\[
\hat y = b + w^\top \zeta
\]

Use scikit-learn linear regression, ridge, or lasso.

Per your notes, the loss/update story is:

\[
J(w)=\|y-\hat y\|^2
\]

and update only the classical model weights. No backprop through the circuit. No train loop for quantum parameters. Page 2 makes that explicit.

So for the coding agent:

- precompute `Z_train`, `Z_test`
- where each row is the augmented feature vector \(\zeta(x)\)
- fit `Ridge` or `LinearRegression` on `Z_train, y_train`
- predict on `Z_test`

## Minimal coding instructions for the agent

Implement these functions in one notebook section:

```python
def phi_single(x_row, train_stats):
    # returns [phi1, phi2, phi3, phi4]
    # phi1 = zscore(x1)
    # phi2 = sign(x2) * log(1 + abs(x2))
    # phi3 = x3
    # phi4 = x4
```

```python
@qml.qnode(dev)
def quantum_feature_map(x_row, phi_row):
    # start in |0000>
    # B(x): for i in 0..3:
    #   RY(phi_i)
    #   RZ(phi_i)
    # A(x):
    #   add single-qubit Z phase terms
    #   add pairwise ZZ interactions using x_i * x_j
    # return [<Z1>, <Z2>, <Z3>, <Z1 Z3>]
```

```python
def build_augmented_features(X, classical_features, train_stats):
    # for each row:
    #   phi_row = phi_single(...)
    #   q_row = quantum_feature_map(x_row, phi_row)
    #   zeta_row = concatenate([classical_features[row], q_row])
    # return full matrix Z
```

Then:

```python
model = Ridge(alpha=...)
model.fit(Z_train, y_train)
y_pred = model.predict(Z_test)
```

## One important correction for the agent

Because your circuit uses only diagonal pairwise \(ZZ\) phases plus \(Z\)-basis expectation measurements, the `RY` layer is essential. Without the `RY` layer, the \(Z\)-diagonal phase terms would not change \(Z\)-basis expectation values in a useful way. So do not remove the `RY` block. Your notes already imply this correctly.

## Clean final spec to hand off

> Implement a fixed 4-qubit Pennylane feature map for each input sample \(x=(x_1,x_2,x_3,x_4)\). First compute transformed scalars \(\phi_1=zscore(x_1)\), \(\phi_2=\log(1+|x_2|)\operatorname{sign}(x_2)\), \(\phi_3=x_3\), \(\phi_4=x_4\). Prepare \(|0000\rangle\). Apply on each qubit \(i\): `RY(phi_i)` then `RZ(phi_i)`. Then apply interaction-phase encoding corresponding to \(A(x)=\exp(i[\sum_i \phi_i Z_i + \sum_{i<j} x_i x_j Z_i Z_j])\), implemented using single-qubit `RZ` gates and pairwise `IsingZZ(x_i*x_j)` gates over all pairs. Measure and return exactly the four quantum features `[<Z0>, <Z1>, <Z2>, <Z0Z2>]`. Concatenate these quantum features with the chosen classical features to form the final feature vector \(\zeta\). Fit only a classical linear model `y_hat = b + w^T zeta` with sklearn; do not train any quantum parameters.
