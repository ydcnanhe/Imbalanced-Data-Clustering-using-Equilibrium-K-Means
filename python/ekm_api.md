# EKM & MiniBatchEKM API Reference

> Detailed documentation for the full-batch `EKM` and the mini-batch `MiniBatchEKM` classes defined in `ekm_sklearn.py`.

---
## Table of Contents
- [EKM (Full Batch)](#ekm-full-batch)
  - [Constructor Parameters](#ekm-constructor-parameters)
  - [Attributes After `fit`](#ekm-attributes-after-fit)
  - [Methods](#ekm-methods)
  - [Behavioral Notes & Edge Cases](#ekm-behavioral-notes--edge-cases)
  - [Usage Examples](#ekm-usage-examples)
- [MiniBatchEKM](#minibatchekm)
  - [Constructor Parameters](#minibatchekm-constructor-parameters)
  - [Attributes After `fit`](#minibatchekm-attributes-after-fit)
  - [Methods](#minibatchekm-methods)
  - [Behavioral Notes & Edge Cases](#minibatchekm-behavioral-notes--edge-cases)
  - [Usage Examples](#minibatchekm-usage-examples)
- [Common Utility Functions](#common-utility-functions)
- [Numerical Stability](#numerical-stability)
- [Versioning](#versioning)

---
## EKM (Full Batch)

The Equilibrium K-Means objective refines standard K-Means by weighting assignments with an adaptive equilibrium correction term.

### EKM Constructor Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_clusters` | int | 3 | Number of clusters. |
| `metric` | str | `'euclidean'` | Distance metric: `'euclidean'` or `'manhattan'` (affects `_pairwise_distance`). |
| `alpha` | float or `'dvariance'` | `0.5` | Smoothing parameter. If `'dvariance'`, computed as `scale / mean(distance^2 to global mean)`. |
| `scale` | float | 2.0 | Scale factor when `alpha='dvariance'`. |
| `max_iter` | int | 500 | Maximum iterations per replicate. |
| `tol` | float | 1e-3 | Relative Frobenius norm tolerance for convergence (`||C-C_old||/||C||`). |
| `n_init` | int | 1 | Number of random restarts (when `init='plus'`). Best objective retained. |
| `init` | `'plus'` or ndarray | `'plus'` | Initialization: k-means++ or user-provided centers (shape `(K, d)`). |
| `random_state` | int or None | None | RNG seed for reproducibility. |
| `use_numba` | bool | False | Enable numba-accelerated weight computation (if available). |
| `numba_threads` | int or None | None | Manually set numba thread count. |

### EKM Attributes After `fit`
| Attribute | Shape / Type | Meaning |
|-----------|--------------|---------|
| `cluster_centers_` | `(K, d)` | Final centers from best restart. |
| `labels_` | `(n,)` | Hard cluster assignments (argmin distances). |
| `n_iter_` | int | Iterations used in best replicate. |
| `objective_` | float | Approximate objective (sum of equilibrium expected distances). |
| `alpha_` | float | Resolved alpha (numeric). |
| `D_` | `(n, K)` | Post-fit distances (not squared). |
| `W_` | `(n, K)` | Equilibrium weights matrix. Rows may not sum to 1 (corrects bias). |
| `U_` | `(n, K)` | Soft assignment probabilities (row-normalized). |

### EKM Methods
| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `fit(X, y=None)` | Runs full-batch optimization across `n_init` restarts. |
| `predict` | `predict(X)` | Returns hard labels for new points. |
| `fit_predict` | `fit_predict(X, y=None)` | Convenience = `fit` + return `labels_`. |
| `transform` | `transform(X)` | Returns distance matrix to centers. |
| `fit_transform` | `fit_transform(X, y=None)` | Runs `fit` then returns cached `D_`. |
| `membership` | `membership(X)` | Returns soft assignment probabilities (numerically stable). |
| `fit_membership` | `fit_membership(X, y=None)` | Fit then return `U_`. |

#### Method Notes
- `membership` uses row-wise min shift before exponentiation to avoid underflow.
- `W_` vs `U_`: `U_` is normalized softmax; `W_` includes the equilibrium correction:  
  `W = U * (1 - alpha * (D2 - J_i))` with `J_i = Σ_k U_ik * D2_ik`.
- Degenerate row (all zero weights) fallback: assign 1 to nearest center.

### EKM Behavioral Notes & Edge Cases
- If a cluster center receives effectively zero total weight, it may drift slowly; consider multiple `n_init` for robustness.
- When `alpha` is large, weights concentrate—risk of singular updates; adjust `scale` or use lower `alpha`.
- Manhattan metric performance may be slower (no numba special-case distance). Centers are still updated by weighted mean (not geometric median), which is an approximation under L1.

### EKM Usage Examples
```python
from ekm_sklearn import EKM
import numpy as np
X = np.random.randn(1000, 8)
model = EKM(n_clusters=5, alpha='dvariance', scale=2.0, max_iter=300, n_init=3, use_numba=True, random_state=42)
model.fit(X)
print(model.cluster_centers_.shape, model.objective_)
proba = model.membership(X[:5])
```

---
## MiniBatchEKM
Mini-batch variant supporting accumulation (statistically closer to full batch) and online updates (faster adaptation).

### MiniBatchEKM Constructor Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_clusters` | int | 3 | Number of clusters. |
| `metric` | str | `'euclidean'` | Distance metric. |
| `alpha` | float or `'dvariance'` | `0.5` | If `'dvariance'`, estimated from warm-up subset (`init_size` or heuristic). |
| `scale` | float | 2.0 | Scale when auto-computing alpha. |
| `batch_size` | int | 256 | Mini-batch size. |
| `max_epochs` | int | 10 | Number of full passes over data. |
| `init` | `'plus'` or ndarray | `'plus'` | Initialization strategy. |
| `init_size` | int or None | None | Warm-up subset size for alpha when using `'dvariance'`. |
| `shuffle` | bool | True | Shuffle order each epoch. |
| `learning_rate` | float or None | None | If `None`: accumulation mode; else online convex blend per batch. |
| `tol` | float | 1e-3 | Relative center movement stopping criterion (per epoch). |
| `reassignment_ratio` | float | 0.0 | Threshold (fraction of batch) below which cluster update may be skipped (online mode). |
| `reassign_patience` | int | 3 | Consecutive weak batches before cluster reassignment (online). |
| `verbose` | int | 1 | Print progress every `print_every` epochs if >0. |
| `monitor_size` | int or None | 1024 | Subset size for approximate objective monitoring. |
| `print_every` | int | 1 | Epoch interval for logging. |
| `use_numba` | bool | False | Enable numba acceleration. |
| `numba_threads` | int or None | None | Numba thread count. |
| `random_state` | int or None | None | RNG seed. |

### MiniBatchEKM Attributes After `fit`
| Attribute | Shape / Type | Meaning |
|-----------|--------------|---------|
| `cluster_centers_` | `(K, d)` | Learned centers. |
| `alpha_` | float | Resolved alpha. |
| `n_epochs_` | int | Epochs actually run (may stop early). |
| `counts_` | `(K,)` | Accumulated effective weights per cluster (accumulation mode). |
| `sums_` | `(K, d)` | Accumulated weighted sums (accumulation mode). |
| `D_` | `(n, K)` | Distances (computed at end of fit). |
| `W_` | `(n, K)` | Final equilibrium weights on full data (post-fit pass). |
| `U_` | `(n, K)` | Final soft assignments on full data. |
| `objective_approx_` | list[float] | Approximate objective values per epoch (monitor subset). |

### MiniBatchEKM Methods
| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `fit(X, y=None)` | Train over multiple epochs. |
| `partial_fit` | `partial_fit(X_batch, y=None)` | Incremental update from a single batch/stream. Initializes if first call. |
| `predict` | `predict(X)` | Hard labels for new points. |
| `transform` | `transform(X)` | Distance matrix. |
| `membership` | `membership(X)` | Soft assignment probabilities (stable). |
| `fit_predict` | `fit_predict(X, y=None)` | Fit then predict. |
| `fit_membership` | `fit_membership(X, y=None)` | Fit then return `U_`. |

### MiniBatchEKM Behavioral Notes & Edge Cases
- Accumulation mode is deterministic given shuffle order and uses running sufficient statistics. 
- Online mode may require tuning `learning_rate`; too high can cause oscillations.
- Reassignment triggers only after `reassign_patience` consecutive weak batches (online mode).
- `monitor_size=None` uses full data for objective approximation (slower, but exact approximation).
- If a batch has zero effective weight for a cluster, its center is not updated; patience counter increments.

### MiniBatchEKM Usage Examples
```python
from ekm_sklearn import MiniBatchEKM
import numpy as np
X = np.random.randn(20000, 16)
mb = MiniBatchEKM(n_clusters=6, alpha='dvariance', scale=2.0, batch_size=512,
                  max_epochs=20, learning_rate=None, monitor_size=2048, verbose=1)
mb.fit(X)
print(mb.n_epochs_, mb.cluster_centers_.shape)
stream = MiniBatchEKM(n_clusters=6, alpha='dvariance', scale=2.0, batch_size=512)
for chunk in np.array_split(X, 40):
    stream.partial_fit(chunk)
labels = stream.predict(X[:1000])
```

---
## Common Utility Functions
- `_pairwise_distance(X, Y=None, metric)`: Backend distance dispatch (euclidean or manhattan). Returns unsquared distances.
- `_kmeans_plus_init(X, K, metric)`: k-means++ seeding.
- `calc_weight(D2, alpha)`: Computes equilibrium weights given squared distances (expects already stabilized if needed).

## Numerical Stability
- All membership and weight computations shift each row by its minimum squared distance before exponentiation.
- Small epsilon (`np.finfo(float).eps`) guards denominator divisions.
- Degenerate weight rows fallback to a hard assignment on nearest center.

## Versioning
- Initial public variant created: 2025-09 (see file header for author and contact).

---
For additional conceptual details, refer to the cited papers in the main README. Feel free to extend this document with further internal notes or experimental variants.
