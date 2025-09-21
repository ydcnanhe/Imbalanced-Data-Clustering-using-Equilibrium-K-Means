"""
High-dimensional imbalanced benchmark (Dirichlet mixture) comparing sklearn KMeans vs EKM.

Features:
  - Generates synthetic data with K_true clusters, using a skewed Dirichlet to produce imbalance.
  - Supports arbitrary dimensionality (default 20).
  - Reports per-run: ARI, NMI, Silhouette (optional), inertia / final SSE, EKM objective, timing.
  - Aggregates mean/std and produces boxplots (if matplotlib installed).

Example (Windows cmd):
  cd python
  python benchmark_dirichlet_highdim.py --n-samples 8000 --n-clusters 8 --n-runs 40 --dim 20 --seed 42 --plot --save-csv dirichlet_highdim.csv

Silhouette can be expensive (O(n^2)); enable with --silhouette and optionally subsample.
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from ekm_sklearn import EKM

try:
    import matplotlib.pyplot as plt  # type: ignore
    _MPL = True
except Exception:  # pragma: no cover
    _MPL = False

@dataclass
class RunStats:
    seed: int
    ari_km: float
    ari_ekm: float
    nmi_km: float
    nmi_ekm: float
    sil_km: float | None
    sil_ekm: float | None
    sse_km: float
    sse_ekm: float
    obj_ekm: float
    time_km: float
    time_ekm: float

# ------------- data generation ------------- #

def gen_dirichlet_mixture(n_samples: int, k_true: int, dim: int, seed: int, imbalance: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    # cluster weights via Dirichlet with one boosted alpha
    alpha = np.ones(k_true)
    alpha[0] = imbalance
    weights = rng.dirichlet(alpha)
    counts = rng.multinomial(n_samples, weights)
    centers = rng.uniform(-8, 8, size=(k_true, dim))
    X_parts = []
    y_parts = []
    for k in range(k_true):
        if counts[k] == 0:
            continue
        scale = rng.uniform(0.4, 1.8)
        cov = np.eye(dim) * (scale ** 2)
        Xk = rng.multivariate_normal(centers[k], cov, size=counts[k])
        X_parts.append(Xk)
        y_parts.append(np.full(counts[k], k, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]

# ------------- main single run ------------- #

def run_once(X: np.ndarray, y: np.ndarray, K: int, seed: int, compute_sil: bool, sil_subsample: int) -> RunStats:
    # KMeans (sklearn)
    t0 = time.perf_counter()
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels_km = km.fit_predict(X)
    t1 = time.perf_counter()
    # EKM
    t2 = time.perf_counter()
    ekm = EKM(n_clusters=K, alpha='dvariance', scale=2.0, n_init=5, random_state=seed)
    labels_ekm = ekm.fit_predict(X)
    t3 = time.perf_counter()

    def maybe_sil(data, labels):
        try:
            if len(np.unique(labels)) <= 1:
                return None
            return float(silhouette_score(data, labels, metric='euclidean'))
        except Exception:
            return None

    if compute_sil:
        if X.shape[0] > sil_subsample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=sil_subsample, replace=False)
            Xs = X[idx]
            ls_km = labels_km[idx]
            ls_ekm = labels_ekm[idx]
        else:
            Xs = X
            ls_km = labels_km
            ls_ekm = labels_ekm
        sil_km = maybe_sil(Xs, ls_km)
        sil_ekm = maybe_sil(Xs, ls_ekm)
    else:
        sil_km = sil_ekm = None

    ari_km = adjusted_rand_score(y, labels_km)
    ari_ekm = adjusted_rand_score(y, labels_ekm)
    nmi_km = normalized_mutual_info_score(y, labels_km)
    nmi_ekm = normalized_mutual_info_score(y, labels_ekm)

    # SSE (inertia) for KMeans is in km.inertia_. For EKM compute directly.
    sse_km = float(km.inertia_)
    # distance squared to nearest center
    from sklearn.metrics.pairwise import euclidean_distances
    D2 = euclidean_distances(X, ekm.cluster_centers_) ** 2
    sse_ekm = float(np.min(D2, axis=1).sum())

    return RunStats(
        seed=seed,
        ari_km=ari_km,
        ari_ekm=ari_ekm,
        nmi_km=nmi_km,
        nmi_ekm=nmi_ekm,
        sil_km=sil_km,
        sil_ekm=sil_ekm,
        sse_km=sse_km,
        sse_ekm=sse_ekm,
        obj_ekm=float(ekm.objective_),
        time_km=t1 - t0,
        time_ekm=t3 - t2,
    )

# ------------- aggregation ------------- #

def mean_std(xs: List[float]):
    arr = np.array(xs, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

# ------------- plotting ------------- #

def plot_box(results: List[RunStats], args):
    if not _MPL:
        print('[Warn] matplotlib not installed; skipping plots.')
        return
    import matplotlib.pyplot as plt
    ari_km = [r.ari_km for r in results]
    ari_ekm = [r.ari_ekm for r in results]
    nmi_km = [r.nmi_km for r in results]
    nmi_ekm = [r.nmi_ekm for r in results]
    sse_km = [r.sse_km for r in results]
    sse_ekm = [r.sse_ekm for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].boxplot([ari_km, ari_ekm], labels=['KMeans','EKM'], patch_artist=True)
    axes[0].set_title('ARI')
    axes[1].boxplot([nmi_km, nmi_ekm], labels=['KMeans','EKM'], patch_artist=True)
    axes[1].set_title('NMI')
    axes[2].boxplot([sse_km, sse_ekm], labels=['KMeans','EKM'], patch_artist=True)
    axes[2].set_title('SSE (lower better)')
    fig.suptitle('High-Dim Dirichlet Imbalanced Benchmark')
    fig.tight_layout()
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=200)
        print(f'[Plot] saved figure to {args.save_plot}')
    if args.plot:
        plt.show()

# ------------- main ------------- #

def run(args):
    X, y = gen_dirichlet_mixture(args.n_samples, args.n_clusters, args.dim, args.seed, args.imbalance)
    seeds = [args.seed + i for i in range(args.n_runs)]
    results: List[RunStats] = []
    for s in seeds:
        r = run_once(X, y, args.n_clusters, s, args.silhouette, args.silhouette_subsample)
        results.append(r)
        if args.verbose:
            print(f"seed {s:4d} | ARI km {r.ari_km:.4f} ekm {r.ari_ekm:.4f} | NMI km {r.nmi_km:.4f} ekm {r.nmi_ekm:.4f} | SSE km {r.sse_km:.1f} ekm {r.sse_ekm:.1f} | time km {r.time_km*1e3:.1f}ms ekm {r.time_ekm*1e3:.1f}ms")

    # aggregation
    def agg_pair(a, b, label, larger_better=True):
        ma, sa = mean_std(a); mb, sb = mean_std(b)
        wins_a = sum(1 for x, y in zip(a, b) if (x > y if larger_better else x < y))
        wins_b = sum(1 for x, y in zip(a, b) if (y > x if larger_better else y < x))
        direction = 'higher' if larger_better else 'lower'
        print(f"{label:<8} KMeans {ma:.4f}±{sa:.4f} | EKM {mb:.4f}±{sb:.4f} | wins km/ekm {wins_a}/{wins_b} ({direction} better)")

    print('\n==== Summary ({} runs, dim={}, K={}) ===='.format(len(results), args.dim, args.n_clusters))
    agg_pair([r.ari_km for r in results], [r.ari_ekm for r in results], 'ARI', True)
    agg_pair([r.nmi_km for r in results], [r.nmi_ekm for r in results], 'NMI', True)
    agg_pair([r.sse_km for r in results], [r.sse_ekm for r in results], 'SSE', False)
    mk, sk = mean_std([r.time_km for r in results]); me, se = mean_std([r.time_ekm for r in results])
    print(f"Time    KMeans {mk*1e3:.2f}±{sk*1e3:.2f} ms | EKM {me*1e3:.2f}±{se*1e3:.2f} ms")
    if args.silhouette:
        sil_km = [r.sil_km for r in results if r.sil_km is not None]
        sil_ekm = [r.sil_ekm for r in results if r.sil_ekm is not None]
        if sil_km and sil_ekm:
            agg_pair(sil_km, sil_ekm, 'SIL', True)
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed','ari_km','ari_ekm','nmi_km','nmi_ekm','sil_km','sil_ekm','sse_km','sse_ekm','obj_ekm','time_km','time_ekm'])
            for r in results:
                w.writerow([r.seed,r.ari_km,r.ari_ekm,r.nmi_km,r.nmi_ekm,r.sil_km,r.sil_ekm,r.sse_km,r.sse_ekm,r.obj_ekm,r.time_km,r.time_ekm])
            print(f"[CSV] wrote per-run stats -> {args.save_csv}")
    if args.plot:
        plot_box(results, args)

# ------------- CLI ------------- #

def parse_args():
    ap = argparse.ArgumentParser(description='High-dimensional Dirichlet imbalance benchmark: KMeans vs EKM')
    ap.add_argument('--n-samples', type=int, default=6000, help='Total samples')
    ap.add_argument('--n-clusters', type=int, default=6, help='True & target clusters')
    ap.add_argument('--dim', type=int, default=20, help='Feature dimensionality')
    ap.add_argument('--n-runs', type=int, default=30, help='Monte Carlo repetitions (different seeds)')
    ap.add_argument('--seed', type=int, default=42, help='Base random seed')
    ap.add_argument('--imbalance', type=float, default=8.0, help='Dirichlet boost factor for first cluster')
    ap.add_argument('--silhouette', action='store_true', help='Compute silhouette (may be slow)')
    ap.add_argument('--silhouette-subsample', type=int, default=4000, help='Subsample size for silhouette if dataset bigger')
    ap.add_argument('--plot', action='store_true', help='Generate boxplots if matplotlib available')
    ap.add_argument('--save-plot', type=str, default='', help='Save boxplot figure to file')
    ap.add_argument('--save-csv', type=str, default='', help='Write per-run CSV stats')
    ap.add_argument('--verbose', action='store_true', help='Print per-run details')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
