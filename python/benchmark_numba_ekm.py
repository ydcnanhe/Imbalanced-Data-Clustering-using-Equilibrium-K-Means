import time
import statistics
import numpy as np
from ekm_sklearn import EKM, _pairwise_distance

try:
    from numba import njit  # noqa: F401
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

"""
Benchmark: Compare EKM (full batch) with and without numba acceleration.

Key Points:
- We keep the same random seed & initial centers to ensure comparable work.
- We include an optional JIT warm-up run (excluded from timing) when numba is available.
- We run multiple repeats and report mean / std.
- If numba isn't installed, we still run the non-numba branch and warn the user.
"""

def make_data(n_samples=12000, n_features=16, n_clusters=5, imbalance=True, seed=42):
    rng = np.random.RandomState(seed)
    sizes = []
    if imbalance:
        base = n_samples // (2 * n_clusters)
        # geometric progression of cluster sizes
        for k in range(n_clusters):
            sizes.append(base * (2 ** (n_clusters - k - 1)))
        ssum = sum(sizes)
        # scale to match n_samples
        sizes = [max(5, int(sz * n_samples / ssum)) for sz in sizes]
        # adjust last cluster to hit total
        diff = n_samples - sum(sizes)
        sizes[-1] += diff
    else:
        sizes = [n_samples // n_clusters] * n_clusters
        sizes[-1] += n_samples - sum(sizes)

    centers_true = rng.uniform(-5, 5, size=(n_clusters, n_features))
    X_parts = []
    y = []
    for k, sz in enumerate(sizes):
        cov = np.diag(rng.uniform(0.3, 1.2, size=n_features))
        Xk = rng.multivariate_normal(centers_true[k], cov, size=sz)
        X_parts.append(Xk)
        y.append(np.full(sz, k))
    X = np.vstack(X_parts)
    y = np.concatenate(y)
    return X, y, centers_true


def clone_initial_centers(X, n_clusters, random_state):
    # Use k-means++ style init once, then reuse for fairness
    from ekm_sklearn import _kmeans_plus_init
    return _kmeans_plus_init(X, n_clusters, metric='euclidean')


def time_run(X, init_centers, use_numba, repeats=3, max_iter=200, tol=1e-3):
    durations = []
    objective = None
    labels_digest = None
    for r in range(repeats):
        model = EKM(n_clusters=init_centers.shape[0], alpha='dvariance', scale=2.0,
                    max_iter=max_iter, tol=tol, n_init=1, init=init_centers,
                    random_state=1234 + r, use_numba=use_numba)
        t0 = time.time()
        model.fit(X)
        t1 = time.time()
        durations.append(t1 - t0)
        if objective is None:
            objective = model.objective_
            # quick digest verifies identical labeling pattern when deterministic
            labels_digest = np.bincount(model.labels_, minlength=model.n_clusters)
    return {
        'use_numba': use_numba,
        'durations': durations,
        'mean': statistics.mean(durations),
        'std': statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        'objective': objective,
        'label_hist': labels_digest,
    }


def maybe_warmup(X, init_centers):
    if not HAVE_NUMBA:
        return
    print('[Warmup] Running one unmeasured JIT warm-up (numba).')
    model = EKM(n_clusters=init_centers.shape[0], alpha='dvariance', scale=2.0,
                max_iter=5, tol=1e-2, n_init=1, init=init_centers,
                random_state=999, use_numba=True)
    model.fit(X[: min(2000, len(X))])  # small subset warm-up


def main():
    X, y, centers_true = make_data()
    n_clusters = centers_true.shape[0]
    init_centers = clone_initial_centers(X, n_clusters, random_state=42)

    if HAVE_NUMBA:
        maybe_warmup(X, init_centers)
    else:
        print('[Info] numba not installed; only measuring pure NumPy path.')

    repeats = 5 if HAVE_NUMBA else 3

    res_no = time_run(X, init_centers, use_numba=False, repeats=repeats)
    res_yes = time_run(X, init_centers, use_numba=True, repeats=repeats) if HAVE_NUMBA else None

    print('\n=== EKM Full Batch numba Benchmark ===')
    header = f"{'Variant':15s} {'Mean(s)':>10s} {'Std(s)':>9s} {'Objective':>12s} {'Hist':>20s}" \
             + ("  Durations" )
    print(header)
    print('-'*len(header))
    print(f"{'No numba':15s} {res_no['mean']:10.4f} {res_no['std']:9.4f} {res_no['objective']:12.4f} {str(res_no['label_hist']):>20s}  {res_no['durations']}")
    if res_yes:
        print(f"{'With numba':15s} {res_yes['mean']:10.4f} {res_yes['std']:9.4f} {res_yes['objective']:12.4f} {str(res_yes['label_hist']):>20s}  {res_yes['durations']}")

    if res_yes and res_no['objective'] != res_yes['objective']:
        print('[Warn] Objectives differ. Check determinism or numerical drift.')

    speedup = (res_no['mean'] / res_yes['mean']) if (res_yes and res_yes['mean'] > 0) else None
    if speedup:
        print(f"\nApprox speedup (No numba / With numba): {speedup:.2f}x")

    print('\nDone.')

if __name__ == '__main__':
    main()
