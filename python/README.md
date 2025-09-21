# Imbalanced Data Clustering using Equilibrium K-Means (EKM)

English | [中文说明](./README_zh.md) | API: [English](./ekm_api.md) | [中文](./ekm_api_zh.md)

> A research / experimental implementation of Equilibrium K-Means (EKM) and its Mini-Batch variant for clustering highly imbalanced datasets. Includes benchmarking utilities, parameter sensitivity studies, and optional numba acceleration.

## Contents Overview

| File | Purpose |
|------|---------|
| `ekm_sklearn.py` | Core implementation: full-batch `EKM` and `MiniBatchEKM` classes, helper functions (`_pairwise_distance`, equilibrium weights). Uses sklearn's `kmeans_plusplus` for Euclidean initialization (falls back to internal routine for other metrics). Optional numba acceleration (`use_numba=True`). |
| `example.py` | Quick start demo showing: (1) full-batch EKM with custom init, (2) MiniBatchEKM accumulation training with progress output. |
| `benchmark.py` | Monte Carlo comparison between vanilla `KMeans` (sklearn) and `EKM` on a fixed imbalanced 3-cluster synthetic dataset (ARI & Silhouette). |
| `benchmark_dirichlet_highdim.py` | High-dimensional Dirichlet-imbalanced benchmark (KMeans vs EKM) with ARI/NMI/SSE/Silhouette & optional plots. |
| `benchmark_alphaSweep.py` | Sensitivity sweep of the scale parameter (effective alpha) for `EKM` versus `KMeans`; prints statistics and plots curves. |
| `benchmark_minibatch_compare.py` | Compares full-batch `EKM` vs MiniBatchEKM (accumulation vs online / learning rate) on an imbalanced Gaussian dataset; reports timing + ARI + NMI + objective approximation. |
| `benchmark_numba_ekm.py` | Benchmarks full-batch `EKM` with and without numba acceleration on a higher-dimensional imbalanced dataset; reports mean/std timings and speedup. |
| `requirements.txt` | Minimal dependencies (NumPy, scikit-learn). Optional: install `numba` and `matplotlib` for acceleration and plotting. |
| `ekm_api.md` / `ekm_api_zh.md` | Detailed API reference (English / Chinese) for `EKM` and `MiniBatchEKM`. |

## Algorithm Summary

Equilibrium K-Means (EKM) modifies the K-Means update using a stabilized equilibrium weighting:
1. Compute squared distances $$D^2$$ between points and current centers.
2. Soft assignments via $$U_{ik} \propto e^{-\alpha D^2_{ik}}$$ (row-wise stabilized by shifting minima for numerical safety).
3. Equilibrium weights $$W_{ik} = U_{ik} (1 - \alpha (D^2_{ik} - J_i))$$ with $$J_i = \sum_k U_{ik} D^2_{ik}$$.
4. Update centers using weighted averages under $$W$$; reassign degenerate rows to nearest centers if needed.
5. Objective (approximate form used for monitoring): $$\sum_i J_i$$.

Mini-batch variant supports two update modes:
- Accumulation (default): maintain running sums / weights (approaches full-batch solution). 
- Online: fixed learning rate blending per-batch weighted centroids (faster adaptation, may need tuning). 

Additional features:
- Row-wise shift to prevent underflow in exponentials.
- Empty / low-weight cluster patience & reassignment.
- Optional approximate objective monitoring per epoch.
- Optional numba-accelerated weight computation.

## Installation

From the `python` directory:
```bash
pip install -r requirements.txt
# Optional
pip install numba matplotlib
```
If using Windows & multiple Python versions, call `py -m pip` instead of `pip`.

## Quick Start

```python
import numpy as np
from ekm_sklearn import EKM

X = np.random.randn(500, 4)
model = EKM(n_clusters=3, alpha='dvariance', scale=2.0, max_iter=300, tol=1e-3, use_numba=False, random_state=0)
model.fit(X)
print("Centers:\n", model.cluster_centers_)
print("Labels shape:", model.labels_.shape)
print("Objective:", model.objective_)
```

Mini-batch variant:
```python
from ekm_sklearn import MiniBatchEKM
mb = MiniBatchEKM(n_clusters=3, alpha='dvariance', scale=2.0,
                  batch_size=256, max_epochs=15, learning_rate=None,
                  monitor_size=1024, print_every=1, verbose=1, random_state=0)
mb.fit(X)
print(mb.cluster_centers_)
```

Streaming / incremental use:
```python
mb_stream = MiniBatchEKM(n_clusters=3, alpha='dvariance', scale=2.0, batch_size=128)
for chunk in np.array_split(X, 10):
    mb_stream.partial_fit(chunk)
labels = mb_stream.predict(X)
```

## Using numba Acceleration
Enable by passing `use_numba=True`:
```python
model = EKM(n_clusters=3, alpha='dvariance', scale=2.0, use_numba=True)
model.fit(X)
```
Numba will JIT compile on first use; consider a warm-up call with a small subset before benchmarking.

## Documentation Site (MkDocs)

Build a local documentation site:
```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
mkdocs serve   # preview at http://127.0.0.1:8000
mkdocs build   # generate static site in site/
```

## Benchmarks

1. `benchmark.py`: Repeated ARI & Silhouette comparison `KMeans` vs `EKM` (Monte Carlo).  
2. `benchmark_alphaSweep.py`: Parameter sensitivity for `scale` (influences computed `alpha` when using `'dvariance'`).  
3. `benchmark_minibatch_compare.py`: Timing + ARI/NMI for full-batch vs accumulation vs online mini-batch.  
4. `benchmark_numba_ekm.py`: Timing comparison of `EKM` with vs without numba (reports speedup).  

Run a benchmark (example):
```bash
python benchmark_minibatch_compare.py
python benchmark_numba_ekm.py
```
(Use `py` on Windows if needed.)

## File Details

### `ekm_sklearn.py`
- Initialization: For `metric='euclidean'`, `init='plus'` now calls sklearn's robust `kmeans_plusplus` seeding (benefits from optimized C / parallel paths). For non-Euclidean metrics a safe Python fallback is used.
- [`EKM`](./ekm_api.md#ekm-full-batch): Full-batch algorithm; parameters:
  - `alpha`: numeric or `'dvariance'` (auto-compute via data variance * `scale`).
  - `use_numba`: toggle accelerated weight computation.
  - `n_init`: multiple random restarts (when `init='plus'`).
- [`MiniBatchEKM`](./ekm_api.md#minibatchekm): Adds `batch_size`, `max_epochs`, `learning_rate`, `reassign_patience`, `monitor_size`, `print_every`.
- Helper functions: `_pairwise_distance`, `_kmeans_plus_init`, `calc_weight`.

### `example.py`
Shows immediate usage of both full-batch and mini-batch forms with printed diagnostics.

### `benchmark_minibatch_compare.py`
Reports: time, ARI, NMI, approximate objective, alpha, iterations/epochs, cluster size distribution.

### `benchmark_numba_ekm.py`
Measures timing across repeats for `use_numba=True/False`, includes optional warm-up.

### `benchmark.py`
Monte Carlo benchmark focusing on ARI and Silhouette distributions (boxplots) vs standard KMeans.

### `benchmark_alphaSweep.py`
Explores performance sensitivity to the scaling parameter (`scale`) controlling effective `alpha`.

## Recommended Hyperparameters

| Scenario | Suggestion |
|----------|------------|
| Highly imbalanced, moderate size | `alpha='dvariance', scale=2.0`, `tol=1e-3` |
| Large dataset (memory) | Use `MiniBatchEKM` with accumulation (`learning_rate=None`) and `batch_size` 256–1024 |
| Fast adaptation / streaming | Use online mode (`learning_rate` in [0.05, 0.3]) with `reassign_patience>=3` |
| Numerical instability observed | Ensure row-wise shift (already built-in)|
| Benchmark speed | Enable `use_numba=True` and warm-up |

## Reproducibility
- Set `random_state` for deterministic center initialization.
- Full determinism with numba may vary slightly due to parallel floating-point reductions; compare objectives with a tolerance.

## Citing / Reference
If you use this codebase in academic work, cite the referenced paper(s)
- He Yudong, An Equilibrium Approach to Clustering: Surpassing Fuzzy C-Means on Imbalanced Data, IEEE Transactions on Fuzzy Systems, 2025.
- He Yudong, Imbalanced Data Clustering Using Equilibrium K-Means, arXiv, 2024.

## License
Distributed under GNU GPL v3 (see header comment). Ensure compatibility before integrating into closed-source systems.

## Future Ideas
- Add unit tests for edge cases (empty cluster, extreme imbalance >1000:1).
- Add adaptive learning rate schedule for online mode.
- Provide a C++ / Pybind11 backend for further acceleration.

---
Feel free to open issues / extend scripts for additional experiments.

## Contact
email: yhebh@connect.ust.hk
