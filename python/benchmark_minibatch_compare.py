import time
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from ekm_sklearn import EKM, MiniBatchEKM

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


def make_imbalanced(seed=0):
    rng = np.random.RandomState(seed)
    # Three Gaussian clusters with heavy imbalance
    n_big, n_mid, n_small = 2000, 50, 30
    mu_big = np.array([-5.0, -2.0])
    mu_mid = np.array([0.0, 0.0])
    mu_small = np.array([5.0, 5.0])
    cov_big = np.array([[1.0, 0.0],[0.0, 1.0]])
    cov_mid = np.array([[1.0, 0.0],[0.0, 1.0]])
    cov_small = np.array([[1.0, 0.0],[0.0, 1.0]])
    X_big = rng.multivariate_normal(mu_big, cov_big, size=n_big)
    X_mid = rng.multivariate_normal(mu_mid, cov_mid, size=n_mid)
    X_small = rng.multivariate_normal(mu_small, cov_small, size=n_small)
    X = np.vstack([X_big, X_mid, X_small])
    y = np.hstack([
        np.zeros(n_big, dtype=int),
        np.ones(n_mid, dtype=int),
        np.full(n_small, 2, dtype=int)
    ])
    return X, y

def stable_objective(X, centers, alpha, metric):
    # compute objective consistent with equilibrium weighting definition
    from ekm_sklearn import _pairwise_distance  # reuse implementation
    D2 = _pairwise_distance(X, centers, metric) ** 2
    # row-wise shift for stability
    D2_shift = D2 - D2.min(axis=1, keepdims=True)
    E = np.exp(-alpha * D2_shift)
    denom = np.sum(E, axis=1) + np.finfo(float).eps
    num = np.sum(D2 * E, axis=1)
    J = num / denom
    return float(np.sum(J))


def run_and_report(name, fit_callable, X, y):
    t0 = time.time()
    model = fit_callable()
    t1 = time.time()
    labels = model.predict(X)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    counts = np.bincount(labels, minlength=model.n_clusters)
    obj = stable_objective(X, model.cluster_centers_, model.alpha_, model.metric) if hasattr(model, 'alpha_') else None
    return {
        'name': name,
        'time_sec': t1 - t0,
        'ARI': ari,
        'NMI': nmi,
        'objective': obj,
        'cluster_sizes': counts,
        'epochs_or_iters': getattr(model, 'n_epochs_', getattr(model, 'n_iter_', None)),
        'alpha': getattr(model, 'alpha_', None)
    }


def main():
    X, y = make_imbalanced(seed=42)
    n_clusters = 3

    # Full batch EKM
    def full_batch():
        m = EKM(n_clusters=n_clusters, alpha='dvariance', scale=2.0, max_iter=300, tol=1e-3, n_init=1, init='plus', random_state=42, use_numba=False)
        m.fit(X)
        return m

    # MiniBatch accumulation
    def minibatch_acc():
        m = MiniBatchEKM(n_clusters=n_clusters, alpha='dvariance', scale=2.0,
                         batch_size=256, max_epochs=30, init='plus', init_size=500,
                         shuffle=True, learning_rate=None, tol=1e-3,
                         reassignment_ratio=0.0, reassign_patience=3, verbose=0,
                         monitor_size=512, print_every=5,
                         use_numba=False, random_state=42)
        m.fit(X)
        return m

    # MiniBatch online (fixed learning rate)
    def minibatch_online():
        m = MiniBatchEKM(n_clusters=n_clusters, alpha='dvariance', scale=2.0,
                         batch_size=256, max_epochs=30, init='plus', init_size=500,
                         shuffle=True, learning_rate=0.2, tol=1e-3,
                         reassignment_ratio=0.01, reassign_patience=3,verbose=0,
                         monitor_size=512, print_every=5,
                         use_numba=False, random_state=42)
        m.fit(X)
        return m

    results = [
        run_and_report('EKM-full', full_batch, X, y),
        run_and_report('MiniBatch-Accum', minibatch_acc, X, y),
        run_and_report('MiniBatch-Online', minibatch_online, X, y)
    ]

    # Pretty print
    print("\n=== Comparison Results ===")
    header = f"{'Method':18s} {'Time(s)':>8s} {'ARI':>7s} {'NMI':>7s} {'Objective':>12s} {'Alpha':>9s} {'Epoch/Iter':>10s}  Cluster Sizes"
    print(header)
    print('-'*len(header))
    for r in results:
        print(f"{r['name']:18s} {r['time_sec']:8.3f} {r['ARI']:7.3f} {r['NMI']:7.3f} {r['objective']:12.4f} {r['alpha']:9.4f} {str(r['epochs_or_iters']):>10s}  {r['cluster_sizes']}")

    # Optional scatter plot colored by method's labels (use full batch) and centers
    if _HAVE_PLT:
        import matplotlib.pyplot as plt
        colors = ['tab:blue','tab:orange','tab:green']
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        for ax, r in zip(axes, results):
            ax.set_title(r['name'])
            # recompute labels to ensure alignment (already have) but re-use
            # Retrieve model via rerun (quick) to get centers easily for display
            # Using earlier model reference would require storing it; here we just re-fit quickly for clarity
            # (Small dataset so overhead negligible.)
            pass
        # Instead, reuse stored centers by re-fitting once more is unnecessary; skip for brevity.
        # Show distribution of true labels with full batch centers overlay
        axes[0].scatter(X[:,0], X[:,1], c=y, s=10, cmap='viridis', alpha=0.6)
        axes[0].scatter(results[0]['cluster_sizes'].argmax(), results[0]['cluster_sizes'].argmax(), alpha=0)  # dummy
        axes[0].set_title('True labels')
        # Show predicted labels for each method
        # Recompute predictions quickly:
        fb = full_batch(); la = minibatch_acc(); lo = minibatch_online()
        for ax, model, title in zip(axes, [fb, la, lo], ['EKM-full','MiniBatch-Accum','MiniBatch-Online']):
            labels = model.predict(X)
            ax.scatter(X[:,0], X[:,1], c=labels, s=10, cmap='viridis', alpha=0.6)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='red', s=60, marker='x')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
