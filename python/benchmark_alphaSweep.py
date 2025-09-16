import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from ekm_sklearn import EKM  # 我们之前实现的 EKM 类
# ------------------
# Benchmark 参数
# ------------------
N_REPEATS = 20   # Monte Carlo 实验次数
N_SAMPLES = [2000, 50, 30]  # 不平衡簇样本
CENTERS = [(-5, -2), (0, 0), (5, 5)]
STD = [1.0, 1.0, 1.0]
N_CLUSTERS = 3
# 设置 α 参数扫描区间
scale_list = [0.1, 0.5, 1.0,1.5,2.0,2.5, 3.0, 4.0] # scale = 2 is used in the paper
results_ekm = {scale: {"ARI": [], "Silhouette": []} for scale in scale_list}
results_km = {"ARI": [], "Silhouette": []}
# ------------------
# Monte Carlo 实验
# ------------------
for seed in range(N_REPEATS):
    # 1. 生成数据
    X, y_true = make_blobs(
        n_samples=N_SAMPLES, centers=CENTERS,
        cluster_std=STD, random_state=seed
    )
    # 2. baseline: KMeans
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
    labels_km = km.fit_predict(X)
    results_km["ARI"].append(adjusted_rand_score(y_true, labels_km))
    results_km["Silhouette"].append(silhouette_score(X, labels_km))
    # 3. 不同 α 的 EKM
    for scale in scale_list:
        ekm = EKM(n_clusters=N_CLUSTERS, distance='sqeuclidean',
                  alpha='dvariance',scale=scale, n_init=10, random_state=seed)
        labels_ekm = ekm.fit_predict(X)
        results_ekm[scale]["ARI"].append(adjusted_rand_score(y_true, labels_ekm))
        results_ekm[scale]["Silhouette"].append(silhouette_score(X, labels_ekm))
# ------------------
# 整理结果
# ------------------
def stats(arr): return np.mean(arr), np.std(arr)
print("\n=== Benchmark Results ({} runs) ===".format(N_REPEATS))
print("KMeans ARI      : {:.3f} ± {:.3f}".format(*stats(results_km["ARI"])))
print("KMeans Silhouette: {:.3f} ± {:.3f}".format(*stats(results_km["Silhouette"])))
for scale in scale_list:
    m_ari, s_ari = stats(results_ekm[scale]["ARI"])
    m_sil, s_sil = stats(results_ekm[scale]["Silhouette"])
    print(f"EKM[scale={scale}] ARI: {m_ari:.3f} ± {s_ari:.3f}, Silhouette: {m_sil:.3f} ± {s_sil:.3f}")
# ------------------
# 绘制 α 参数敏感性曲线
# ------------------
# 只绘制数值型的 α（忽略 'dvariance'）
num_scales = [a for a in scale_list if isinstance(a, float)]
ari_means = [np.mean(results_ekm[a]["ARI"]) for a in num_scales]
sil_means = [np.mean(results_ekm[a]["Silhouette"]) for a in num_scales]
plt.figure(figsize=(10,5))
plt.plot(num_scales, ari_means, marker='o', label="EKM ARI")
plt.plot(num_scales, sil_means, marker='s', label="EKM Silhouette")
plt.axhline(np.mean(results_km["ARI"]), color='red', linestyle='--', label="KMeans ARI (baseline)")
plt.axhline(np.mean(results_km["Silhouette"]), color='green', linestyle='--', label="KMeans Silhouette (baseline)")
plt.xlabel("Alpha parameter")
plt.ylabel("Score")
plt.title(f"EKM Sensitivity to Alpha (mean over {N_REPEATS} runs)")
plt.legend()
plt.show()