import numpy as np
def ekm(X, K, 
        Distance='sqeuclidean', 
        alpha=0.5, 
        MaxIter=500, 
        Eta=1e-3, 
        Replicates=1, 
        Start='plus'):
    """
    Enhanced K-means (Equilibrium K-means, EKM).
    Parameters
    ----------
    X : ndarray of shape (N, P)
        Data matrix containing N samples with P features.
    K : int
        Number of clusters.
    Distance : str, default 'sqeuclidean'
        'sqeuclidean' or 'cosine'
    alpha : float or str, default 0.5
        Smoothing parameter or 'dvariance'
    MaxIter : int, default 500
        Maximum number of iterations.
    Eta : float, default 1e-3
        Tolerance for convergence.
    Replicates : int, default 1
        Number of replications. The replication with lowest objective is retained.
    Start : str or ndarray, default 'plus'
        Initialization method ('plus') or K x P ndarray for user-defined centroids.
    Returns
    -------
    idx : ndarray of shape (N,)
        Cluster indices per sample.
    C : ndarray of shape (K, P)
        Centroids of clusters.
    numit : int
        Number of iterations performed.
    W : ndarray of shape (N, K)
        Weights per sample per cluster.
    U : ndarray of shape (N, K)
        Membership values.
    sumd : ndarray of shape (K,)
        Sum of distances within each cluster.
    D : ndarray of shape (N, K)
        Distance matrix.
    J : float
        Objective function value.
    """
    N, P = X.shape
    # Choose distance function
    if Distance == 'sqeuclidean':
        dist = sqeuclidean
    elif Distance == 'cosine':
        dist = cosine
    else:
        raise ValueError("Unsupported distance metric.")
    # Handle alpha
    if isinstance(alpha, str):
        if alpha == 'dvariance':
            alpha = 2.0 / np.mean(dist(X, np.mean(X, axis=0).reshape(1, -1)))
        else:
            raise ValueError("Unsupported alpha option.")
    # Replicates handling
    if isinstance(Start, np.ndarray):
        R = Start.shape[0] // K
    else:
        R = Replicates
    J_set = np.zeros(R)
    C_set = np.zeros((R, K, P))
    numit_set = np.zeros(R, dtype=int)
    for r in range(R):
        # initialization
        if isinstance(Start, np.ndarray):
            C = Start[r*K:(r+1)*K, :]
        else:
            if Start == 'plus':
                C = kmeans_plus_init(X, K)
            else:
                raise ValueError("Unsupported initialization method.")
        C_old = C.copy()
        it = 1
        while True:
            D = dist(X, C)
            W = calc_weight(D, alpha)
            # update centroids
            for k in range(K):
                sum_wk = np.sum(W[:, k])
                if sum_wk == 0:
                    sum_wk = np.finfo(float).eps
                C[k, :] = np.dot(W[:, k], X) / sum_wk
            # convergence check
            if np.linalg.norm(C - C_old, 'fro') / np.linalg.norm(C, 'fro') < Eta:
                break
            if it >= MaxIter:
                print(f"Failed to converge in {MaxIter} iterations during replicate {r+1} for EKM with {K} clusters")
                break
            else:
                C_old = C.copy()
                it += 1
        D = dist(X, C)
        J = np.sum(np.sum(D * np.exp(-alpha*D), axis=1) / (np.sum(np.exp(-alpha*D), axis=1) + np.finfo(float).eps))
        J_set[r] = J
        C_set[r] = C
        numit_set[r] = it
    best_id = np.argmin(J_set)
    J = J_set[best_id]
    C = C_set[best_id]
    numit = numit_set[best_id]
    D = dist(X, C)
    idx = np.argmin(D, axis=1)
    sumd = np.array([np.sum(D[idx == k, k]) for k in range(K)])
    W = calc_weight(D, alpha)
    U = np.exp(-alpha * D) / (np.sum(np.exp(-alpha * D), axis=1, keepdims=True) + np.finfo(float).eps)
    return idx, C, numit, W, U, sumd, D, J
def kmeans_plus_init(X, K):
    """K-means++ initialization"""
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
    """Squared Euclidean distance"""
    return np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)
def cosine(X, C):
    """Cosine distance"""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + np.finfo(float).eps)
    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True)[:, None] + np.finfo(float).eps)
    return 1 - np.dot(X_norm, C_norm.T)
def calc_weight(D, alpha):
    """Weight calculation"""
    N, K = D.shape
    J = np.sum(D * np.exp(-alpha * D), axis=1) / (np.sum(np.exp(-alpha * D), axis=1) + np.finfo(float).eps)
    W = np.exp(-alpha * D) / (np.sum(np.exp(-alpha * D), axis=1, keepdims=True) + np.finfo(float).eps) \
        * (1 - alpha * (D - J[:, None]))
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    for i in zero_idx:
        pos = np.argmin(D[i])
        W[i] = np.eye(1, K, pos)
    return W