import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
# Optional: numba acceleration for weight computation
try:
    from numba import njit, prange, set_num_threads
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
# ---------------- helper functions ---------------- #
def _pairwise_distance(X, Y=None, metric="euclidean"):
    if metric == "euclidean":
        return euclidean_distances(X, Y, squared=False)
    elif metric == "manhattan":
        return manhattan_distances(X, Y)
    else:
        raise ValueError(f"Unsupported distance: {metric}")
def _kmeans_plus_init(X, K, metric="euclidean"):
    N, P = X.shape
    C = np.empty((K, P))
    idx = np.random.randint(N)
    C[0] = X[idx]
    for k in range(1, K):
        D2 = np.min(_pairwise_distance(X, C[:k], metric)**2, axis=1)
        probs = D2 / np.sum(D2)
        idx = np.random.choice(N, p=probs)
        C[k] = X[idx]
    return C

def calc_weight(D2, alpha):
    N, K = D2.shape
    E = np.exp(-alpha * D2)
    U = E / np.sum(E, axis=1, keepdims=True)
    W = U * (1 - alpha * (D2 - np.sum(D2 * U, axis=1, keepdims=True)))
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    for i in zero_idx:
        pos = np.argmin(D2[i])
        W[i] = np.eye(1, K, pos)
    return W

# ---------------- optional accelerated weight (numba) ---------------- #
if _NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _calc_weight_numba(D2, alpha):
        N, K = D2.shape
        W = np.empty((N, K), dtype=np.float64)
        for i in prange(N):
            # compute exp(-alpha * D2[i]) and sums
            sum_e = 0.0
            sum_d2e = 0.0
            # store to reuse
            # Numba requires fixed-size, so allocate per-row array
            row_e = np.empty(K, dtype=np.float64)
            for k in range(K):
                e = np.exp(-alpha * D2[i, k])
                row_e[k] = e
                sum_e += e
                sum_d2e += D2[i, k] * e
            denom = sum_e
            J_i = sum_d2e / denom
            row_sumW = 0.0
            for k in range(K):
                w = (row_e[k] / denom) * (1.0 - alpha * (D2[i, k] - J_i))
                W[i, k] = w
                row_sumW += w
            if row_sumW == 0.0:
                # hard-assign to closest center
                best = D2[i, 0]
                pos = 0
                for k in range(1, K):
                    if D2[i, k] < best:
                        best = D2[i, k]
                        pos = k
                for k in range(K):
                    W[i, k] = 0.0
                W[i, pos] = 1.0
        return W
class EKM(BaseEstimator, ClusterMixin):
    """
    Information
    ----------
    Equilibrium K-means (EKM) - a robust variant of K-means for imbalanced data clustering.
    Version 2
    Created at 16, Sep, 2025
    Last modified at 19, Sep, 2025
    Author: Yudong He
    Email: yhebh@connect.ust.hk
    Update: 
    1. Fixed underflow issue in weight and membership computation via per-row shifting.
    2. Added optional numba acceleration for weight computation.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    metric : str, default 'euclidean'
        Distance metric: 'euclidean' or 'manhattan'.
    alpha : float or str, default 0.5
        Smoothing parameter or 'dvariance'.
    scale : float, default 2
        Scaloing the alpha when alpha = 'dvariance'
    max_iter : int, default 500
        Maximum number of iterations.
    tol : float, default 1e-3
        Convergence tolerance.
    n_init : int, default 1
        Number of replicates / runs with different centroid seeds.
    init : str, default 'plus'
        Initialization method: 'plus' (k-means++) or ndarray of initial centers.
    random_state : int, optional
        Seed for reproducibility.

    Reference
    ---------
    He Yudong, An Equilibrium Approach to Clustering: Surpassing Fuzzy C-Means on Imbalanced Data, IEEE Transactions on Fuzzy Systems, 2025.

    He Yudong, Imbalanced Data Clustering Using Equilibrium K-Means, arXiv, 2024.

    Copyright
    ---------
    This software is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt)

    ---------
    """
    def __init__(self, n_clusters=3, metric='euclidean', alpha=0.5, scale=2.0,
                 max_iter=500, tol=1e-3, n_init=1, init='plus', random_state=None,
                 use_numba=False, numba_threads=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.alpha = alpha
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.use_numba = bool(use_numba)
        self.numba_threads = numba_threads

    def _calc_weight_batch(self, D2, alpha):
        # choose accelerated or numpy implementation
        if self.use_numba and _NUMBA_AVAILABLE:
            if self.numba_threads is not None:
                try:
                    set_num_threads(int(self.numba_threads))
                except Exception:
                    pass
            return _calc_weight_numba(D2, alpha)
        return calc_weight(D2, alpha)
    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        N, P = X.shape
        K = self.n_clusters
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == 'dvariance':
                alpha = self.scale / np.mean(_pairwise_distance(X, np.mean(X, axis=0).reshape(1, -1), self.metric)**2)
            else:
                raise ValueError("Unsupported alpha option.")
        best_obj = np.inf
        best_centers = None
        best_labels = None
        best_niter = None
        for r in range(self.n_init):
            # initialization
            if isinstance(self.init, np.ndarray):
                C = self.init.copy()
            elif self.init == 'plus':
                C = _kmeans_plus_init(X, K, self.metric)
            else:
                raise ValueError("Unsupported init method.")
            C_old = C.copy()
            it = 1
            while True:
                D2 = _pairwise_distance(X, C, self.metric)**2
                # underflow-safe via row-wise shift; safe because W depends on exp(D2)
                D2_shift = D2 - D2.min(axis=1, keepdims=True)
                W = self._calc_weight_batch(D2_shift, alpha)
                for k in range(K):
                    sum_wk = np.maximum(np.sum(W[:, k]), np.finfo(float).eps)
                    C[k, :] = np.dot(W[:, k], X) / sum_wk
                if np.linalg.norm(C - C_old, 'fro') / (np.linalg.norm(C, 'fro') + 1e-12) < self.tol:
                    break
                if it >= self.max_iter:
                    print(f"[Warning] Replicate {r+1}: Failed to converge in {self.max_iter} iterations.")
                    break
                else:
                    C_old = C.copy()
                    it += 1
            D2 = _pairwise_distance(X, C, self.metric)**2
            # approximate objective (numerically stable): shift per-row for exp, but use original D2 in numerator
            D2_shift_eval = D2 - D2.min(axis=1, keepdims=True)
            E = np.exp(-alpha * D2_shift_eval)
            U = E / np.sum(E, axis=1, keepdims=True)
            obj = np.sum(U * D2)
            if obj < best_obj:
                best_obj = obj
                best_centers = C
                best_labels = np.argmin(D2, axis=1)
                best_niter = it
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.n_iter_ = best_niter
        self.objective_ = best_obj
        # store extra info:
        self.D_ = _pairwise_distance(X, self.cluster_centers_, self.metric)
        D2_full = self.D_**2
        D2_shift_full = D2_full - D2_full.min(axis=1, keepdims=True)
        self.W_ = self._calc_weight_batch(D2_shift_full, alpha)
        E_full = np.exp(-alpha * D2_shift_full)
        self.U_ = E_full / np.sum(E_full, axis=1, keepdims=True)
        self.alpha_= alpha
        return self
    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        D = _pairwise_distance(X, self.cluster_centers_, self.metric)
        return np.argmin(D, axis=1)
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
    def transform(self, X):
        """Return distance to cluster centers"""
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        return _pairwise_distance(X, self.cluster_centers_, self.metric)
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.D_
    def membership(self, X):
        """membership matrix"""
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        else:
            alpha = self.alpha_
            D2 = self.transform(X)**2
            D2_shift = D2 - D2.min(axis=1, keepdims=True)
            E = np.exp(-alpha * D2_shift)
            U = E / np.sum(E, axis=1, keepdims=True)
            return U
    def fit_membership(self, X, y=None):
        self.fit(X, y)
        return self.U_

# ---------------- MiniBatch EKM ---------------- #
class MiniBatchEKM(BaseEstimator, ClusterMixin):
    """
    Information
    ----------
    Mini-batch variant of Equilibrium K-Means (EKM) for scalable clustering.
    Version 1
    Created at 19, Sep, 2025
    Last modified at 19, Sep, 2025
    Author: Yudong He
    Email: yhebh@connect.ust.hk
    Key features:
    1. Mini-batch processing for scalability.
    2. Underflow-safe membership and weight computations via per-row shifting.
    3. Per-cluster minimum weight threshold to skip noisy updates.
    4. Patient empty-cluster reassignment with a counter instead of single-shot.
    5. Optional numba-parallel weight computation for speed.
    6. Epoch-wise approximate objective monitoring with progress printing.

    Copyright
    ---------
    This software is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt)

    ---------

    """

    def __init__(self, n_clusters=3, metric='euclidean', alpha=0.5, scale=2.0,
                 batch_size=256, max_epochs=10, init='plus', init_size=None,
                 shuffle=True, learning_rate=None, tol=1e-3,
                 reassignment_ratio=0.0, reassign_patience=3,
                 verbose=1, monitor_size=1024, print_every=1,
                 use_numba=False, numba_threads=None,
                 random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.alpha = alpha
        self.scale = scale
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.init = init
        self.init_size = init_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.tol = tol
        self.reassignment_ratio = reassignment_ratio
        self.reassign_patience = int(reassign_patience)
        self.verbose = int(verbose)
        self.monitor_size = None if monitor_size is None else int(monitor_size)
        self.print_every = max(1, int(print_every))
        self.use_numba = bool(use_numba)
        self.numba_threads = numba_threads
        self.random_state = random_state

    # ---------- initialization helpers ---------- #
    def _init_centers(self, X):
        K = self.n_clusters
        if isinstance(self.init, np.ndarray):
            C = self.init.astype(float).copy()
        elif self.init == 'plus':
            C = _kmeans_plus_init(X, K, self.metric).astype(float)
        else:
            raise ValueError("Unsupported init method.")
        return C

    def _init_alpha(self, X):
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == 'dvariance':
                # Warm-start subset for alpha estimation
                n0 = (min(len(X), max(10 * self.n_clusters, self.batch_size))
                      if self.init_size is None else min(len(X), self.init_size))
                idx = np.random.choice(len(X), size=n0, replace=False)
                X0 = X[idx]
                mu = np.mean(X0, axis=0, keepdims=True)
                d2 = _pairwise_distance(X0, mu, self.metric) ** 2
                dv = float(np.mean(d2))
                alpha = self.scale / max(dv, np.finfo(float).eps)
            else:
                raise ValueError("Unsupported alpha option.")
        return float(alpha)

    def _calc_weight_batch(self, D2, alpha):
        # choose accelerated or numpy implementation
        if self.use_numba and _NUMBA_AVAILABLE:
            if self.numba_threads is not None:
                try:
                    set_num_threads(int(self.numba_threads))
                except Exception:
                    pass
            return _calc_weight_numba(D2, alpha)
        return calc_weight(D2, alpha)

    def _approx_objective(self, Xs, C, alpha):
        D2 = _pairwise_distance(Xs, C, self.metric) ** 2
        # underflow-safe soft-min exp via row-wise shift
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        E = np.exp(-alpha * D2_shift)
        denom = np.sum(E, axis=1) + np.finfo(float).eps
        num = np.sum(D2 * E, axis=1)
        J = num / denom
        return float(np.sum(J))

    # ---------- main training loop ---------- #
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        N, P = X.shape
        K = self.n_clusters

        # init centers and alpha
        C = self._init_centers(X)
        alpha = self._init_alpha(X)

        # empty-cluster patience counters
        empty_counts = np.zeros(K, dtype=np.int64)

        prev_C = C.copy()
        self.objective_approx_ = []
        # fix monitor subset for consistent tracking
        if self.monitor_size is None:
            monitor_idx = np.arange(N)
        else:
            ms = min(N, self.monitor_size)
            monitor_idx = rng.choice(N, size=ms, replace=False)

        for epoch in range(1, self.max_epochs + 1):
            if self.shuffle:
                order = rng.permutation(N)
            else:
                order = np.arange(N)

            # initialize accumulators (for accumulation method)
            Nk = np.zeros(K, dtype=float)
            Sk = np.zeros((K, P), dtype=float)

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = order[start:end]
                Xb = X[batch_idx]

                # distances and weights (underflow-safe shift)
                D2 = _pairwise_distance(Xb, C, self.metric) ** 2
                D2_shift = D2 - D2.min(axis=1, keepdims=True)
                W = self._calc_weight_batch(D2_shift, alpha)

                wk_sum = W.sum(axis=0)  # shape (K,)
                abs_wk = np.abs(wk_sum)

                # determine clusters to update based on reassignment ratio
                update_mask = abs_wk > self.reassignment_ratio * Xb.shape[0]

                # update empty counters
                for k in range(K):
                    if update_mask[k]:
                        empty_counts[k] = 0
                    else:
                        empty_counts[k] += 1

                # updates
                if self.learning_rate is None:
                    # accumulation method: zero-out weak clusters to avoid noisy updates
                    if not np.all(update_mask):
                        W_eff = W.copy()
                        W_eff[:, ~update_mask] = 0.0
                        Sk += W_eff.T @ Xb
                        Nk += W_eff.sum(axis=0)
                    else:
                        Sk += W.T @ Xb
                        Nk += wk_sum
                    denom = np.maximum(Nk[:, None], np.finfo(float).eps)
                    C = Sk / denom
                else:
                    # online method with learning rate
                    lr = float(self.learning_rate)
                    for k in range(K):
                        if not update_mask[k]:
                            continue
                        wk = wk_sum[k]
                        xbar_k = (W[:, k][:, None] * Xb).sum(axis=0) / wk
                        C[k] = (1.0 - lr) * C[k] + lr * xbar_k

                # patient empty-cluster reassignment (only meaningful in online mode)
                if self.learning_rate is not None and self.reassign_patience > 0:
                    to_reassign = np.where(empty_counts >= self.reassign_patience)[0]
                    if to_reassign.size > 0:
                        # choose farthest samples per cluster as new centers
                        # approximate: per-column argmax of D2
                        far_idx = np.argmax(D2, axis=0)
                        for k in to_reassign:
                            C[k] = Xb[far_idx[k]]
                            empty_counts[k] = 0  # reset counter after reassignment

            # end epoch: check progress
            delta = np.linalg.norm(C - prev_C, ord='fro') / (np.linalg.norm(C, ord='fro') + 1e-12)
            prev_C = C.copy()

            # monitor approximate objective on fixed subset
            obj_approx = self._approx_objective(X[monitor_idx], C, alpha)
            self.objective_approx_.append(obj_approx)
            if self.verbose and (epoch % self.print_every == 0):
                print(f"[MiniBatchEKM] epoch {epoch}/{self.max_epochs}  delta={delta:.3e}  obj≈{obj_approx:.6g}")

            if delta < self.tol:
                if self.verbose:
                    print(f"[MiniBatchEKM] Converged at epoch {epoch} (delta < tol).")
                break

        # finalize attributes
        self.cluster_centers_ = C
        self.alpha_ = alpha
        self.n_epochs_ = epoch
        self.counts_ = Nk
        self.sums_ = Sk

        # cache distances, W and numerically-stable U
        D = _pairwise_distance(X, C, self.metric)
        D2 = D ** 2
        D2_shift_full = D2 - D2.min(axis=1, keepdims=True)
        self.D_ = D
        self.W_ = self._calc_weight_batch(D2_shift_full, alpha)
        E = np.exp(-alpha * D2_shift_full)
        self.U_ = E / np.sum(E, axis=1, keepdims=True)

        return self

    def partial_fit(self, X_batch, y=None):
        Xb = np.asarray(X_batch, dtype=float)
        if not hasattr(self, "cluster_centers_"):
            # initialize from the first batch
            self.cluster_centers_ = self._init_centers(Xb)
            self.alpha_ = self._init_alpha(Xb)
            K = self.n_clusters
            P = Xb.shape[1]
            self.counts_ = np.zeros(K, dtype=float)
            self.sums_ = np.zeros((K, P), dtype=float)
            self._empty_counts = np.zeros(K, dtype=np.int64)
        else:
            K = self.n_clusters
            if not hasattr(self, "_empty_counts"):
                self._empty_counts = np.zeros(K, dtype=np.int64)

        C = self.cluster_centers_
        alpha = self.alpha_

        D2 = _pairwise_distance(Xb, C, self.metric) ** 2
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        W = self._calc_weight_batch(D2_shift, alpha)

        wk_sum = W.sum(axis=0)
        abs_wk = np.abs(wk_sum)
        # determine clusters to update based on reassignment ratio
        update_mask = abs_wk > self.reassignment_ratio * Xb.shape[0]

        # update empty counters
        for k in range(K):
            if update_mask[k]:
                self._empty_counts[k] = 0
            else:
                self._empty_counts[k] += 1

        if getattr(self, "learning_rate", None) is None:
            if not np.all(update_mask):
                W_eff = W.copy()
                W_eff[:, ~update_mask] = 0.0
                self.sums_ += W_eff.T @ Xb
                self.counts_ += W_eff.sum(axis=0)
            else:
                self.sums_ += W.T @ Xb
                self.counts_ += wk_sum
            denom = np.maximum(self.counts_[:, None], np.finfo(float).eps)
            self.cluster_centers_ = self.sums_ / denom
        else:
            lr = float(self.learning_rate)
            for k in range(K):
                if not update_mask[k]:
                    continue
                wk = wk_sum[k]
                if wk <= 0:
                    continue
                xbar_k = (W[:, k][:, None] * Xb).sum(axis=0) / wk
                C[k] = (1.0 - lr) * C[k] + lr * xbar_k
            self.cluster_centers_ = C

        # patient reassignment
        if getattr(self, "learning_rate", None) is not None and self.reassign_patience > 0:
            to_reassign = np.where(self._empty_counts >= self.reassign_patience)[0]
            if to_reassign.size > 0:
                far_idx = np.argmax(D2, axis=0)
                for k in to_reassign:
                    self.cluster_centers_[k] = Xb[far_idx[k]]
                    self._empty_counts[k] = 0

        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        D = _pairwise_distance(np.asarray(X, dtype=float), self.cluster_centers_, self.metric)
        return np.argmin(D, axis=1)

    def transform(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        return _pairwise_distance(np.asarray(X, dtype=float), self.cluster_centers_, self.metric)

    def membership(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        D2 = self.transform(X) ** 2
        # Underflow-safe U by row-wise shift
        D2_shift = D2 - D2.min(axis=1, keepdims=True)
        alpha = self.alpha_
        E = np.exp(-alpha * D2_shift)
        U = E / (np.sum(E, axis=1, keepdims=True) + np.finfo(float).eps)
        return U

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def fit_membership(self, X, y=None):
        self.fit(X, y)
        return self.U_