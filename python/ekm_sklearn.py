import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
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
class EKM(BaseEstimator, ClusterMixin):
    """
    Information
    ----------
    Equilibrium K-means (EKM) - a robust variant of K-means for imbalanced data clustering.
    Version 1
    Created at Sep, 2025
    Last modified at Sep, 2025
    Author: Yudong He
    Email: yhebh@connect.ust.hk
    
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
                 max_iter=500, tol=1e-3, n_init=1, init='plus', random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.alpha = alpha
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
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
                W = calc_weight(D2, alpha)
                for k in range(K):
                    sum_wk = np.sum(W[:, k])
                    if sum_wk == 0:
                        sum_wk = np.finfo(float).eps
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
            obj = np.sum(np.sum(D2 * np.exp(-alpha*D2), axis=1) / (np.sum(np.exp(-alpha*D2), axis=1) + np.finfo(float).eps))
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
        self.W_ = calc_weight(self.D_**2, alpha)
        self.U_ = np.exp(-alpha * self.D_**2) / (np.sum(np.exp(-alpha * self.D_**2), axis=1, keepdims=True) + np.finfo(float).eps)
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
            U = np.exp(-alpha * D2) / (np.sum(np.exp(-alpha * D2), axis=1, keepdims=True) + np.finfo(float).eps)
            return U
    def fit_membership(self, X, y=None):
        self.fit(X, y)
        return self.U_
    
# ---------------- helper functions ---------------- #
def calc_weight(D2, alpha):
    N, K = D2.shape
    J = np.sum(D2 * np.exp(-alpha * D2), axis=1) / (np.sum(np.exp(-alpha * D2), axis=1) + np.finfo(float).eps)
    W = np.exp(-alpha * D2) / (np.sum(np.exp(-alpha * D2), axis=1, keepdims=True) + np.finfo(float).eps) \
        * (1 - alpha * (D2 - J[:, None]))
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    for i in zero_idx:
        pos = np.argmin(D2[i])
        W[i] = np.eye(1, K, pos)
    return W