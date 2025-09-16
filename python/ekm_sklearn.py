import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
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
    distance : str, default 'sqeuclidean'
        Distance metric: 'sqeuclidean' or 'cosine'.
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
    def __init__(self, n_clusters=3, distance='sqeuclidean', alpha=0.5, scale=2.0,
                 max_iter=500, tol=1e-3, n_init=1, init='plus', random_state=None):
        self.n_clusters = n_clusters
        self.distance = distance
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
        # choose distance function
        if self.distance == 'sqeuclidean':
            dist = sqeuclidean
        elif self.distance == 'cosine':
            dist = cosine
        else:
            raise ValueError("Unsupported distance metric.")
        alpha = self.alpha
        if isinstance(alpha, str):
            if alpha == 'dvariance':
                alpha = self.scale / np.mean(dist(X, np.mean(X, axis=0).reshape(1, -1)))
            else:
                raise ValueError("Unsupported alpha option.")
        J_set = []
        C_set = []
        numit_set = []
        labels_set = []
        for r in range(self.n_init):
            # initialization
            if isinstance(self.init, np.ndarray):
                C = self.init.copy()
            elif self.init == 'plus':
                C = kmeans_plus_init(X, K)
            else:
                raise ValueError("Unsupported init method.")
            C_old = C.copy()
            it = 1
            while True:
                D = dist(X, C)
                W = calc_weight(D, alpha)
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
            D = dist(X, C)
            J = np.sum(np.sum(D * np.exp(-alpha*D), axis=1) / (np.sum(np.exp(-alpha*D), axis=1) + np.finfo(float).eps))
            labels = np.argmin(D, axis=1)
            J_set.append(J)
            C_set.append(C)
            numit_set.append(it)
            labels_set.append(labels)
        best = np.argmin(J_set)
        self.cluster_centers_ = C_set[best]
        self.labels_ = labels_set[best]
        self.n_iter_ = numit_set[best]
        self.inertia_ = J_set[best]
        # store extra info:
        self.D_ = dist(X, self.cluster_centers_)
        self.W_ = calc_weight(self.D_, alpha)
        self.U_ = np.exp(-alpha * self.D_) / (np.sum(np.exp(-alpha * self.D_), axis=1, keepdims=True) + np.finfo(float).eps)
        return self
    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        if self.distance == 'sqeuclidean':
            dist = sqeuclidean
        else:
            dist = cosine
        D = dist(X, self.cluster_centers_)
        return np.argmin(D, axis=1)
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
    def transform(self, X):
        """Return distance to cluster centers"""
        if self.distance == 'sqeuclidean':
            return sqeuclidean(X, self.cluster_centers_)
        else:
            return cosine(X, self.cluster_centers_)
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
# ---------------- helper functions ---------------- #
def kmeans_plus_init(X, K):
    N, P = X.shape
    C = np.empty((K, P))
    idx = np.random.randint(N)
    C[0] = X[idx]
    for k in range(1, K):
        D = np.min(sqeuclidean(X, C[:k]), axis=1)
        probs = D / np.sum(D)
        idx = np.random.choice(N, p=probs)
        C[k] = X[idx]
    return C
def sqeuclidean(X, C):
    return np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)
def cosine(X, C):
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + np.finfo(float).eps)
    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True)[:, None] + np.finfo(float).eps)
    return 1 - np.dot(X_norm, C_norm.T)
def calc_weight(D, alpha):
    N, K = D.shape
    J = np.sum(D * np.exp(-alpha * D), axis=1) / (np.sum(np.exp(-alpha * D), axis=1) + np.finfo(float).eps)
    W = np.exp(-alpha * D) / (np.sum(np.exp(-alpha * D), axis=1, keepdims=True) + np.finfo(float).eps) \
        * (1 - alpha * (D - J[:, None]))
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    for i in zero_idx:
        pos = np.argmin(D[i])
        W[i] = np.eye(1, K, pos)
    return W