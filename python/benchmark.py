import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from ekm_sklearn import EKM  # 之前封装好的类
# ------------------
# Benchmark 参数
# ------------------
N_REPEATS = 30   # Monte Carlo 实验次数
N_SAMPLES = [2000, 50, 30]  # 不平衡簇样本数
CENTERS = [(-5, -2), (0, 0), (5, 5)]
STD = [1.0, 1.0, 1.0]
N_CLUSTERS = 3
# 保存结果
ari_results = {"KMeans": [], "EKM": []}
sil_results = {"KMeans": [], "EKM": []}
# ------------------
# Monte Carlo 实验
# ------------------
for seed in range(N_REPEATS):
    # 1. 生成数据
    X, y_true = make_blobs(
        n_samples=N_SAMPLES, centers=CENTERS, 
        cluster_std=STD, random_state=seed
    )
    # 2. KMeans
    km = KMeans(n_clusters=N_CLUSTERS, n_init=30, random_state=seed)
    labels_km = km.fit_predict(X)
    ari_results["KMeans"].append(adjusted_rand_score(y_true, labels_km))
    sil_results["KMeans"].append(silhouette_score(X, labels_km))
    # 3. EKM
    ekm = EKM(n_clusters=N_CLUSTERS, metric='euclidean',
              alpha='dvariance', n_init=30, random_state=seed)
    labels_ekm = ekm.fit_predict(X)
    ari_results["EKM"].append(adjusted_rand_score(y_true, labels_ekm))
    sil_results["EKM"].append(silhouette_score(X, labels_ekm))
# ------------------
# 统计结果
# ------------------
def stats(arr):
    return np.mean(arr), np.std(arr)
print("\n=== Monte Carlo Benchmark Results ({} runs) ===".format(N_REPEATS))
print("KMeans ARI      : {:.3f} ± {:.3f}".format(*stats(ari_results["KMeans"])))
print("EKM    ARI      : {:.3f} ± {:.3f}".format(*stats(ari_results["EKM"])))
print("KMeans Silhouette: {:.3f} ± {:.3f}".format(*stats(sil_results["KMeans"])))
print("EKM    Silhouette: {:.3f} ± {:.3f}".format(*stats(sil_results["EKM"])))
# ------------------
# 绘制 Boxplot
# ------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# ARI
axs[0].boxplot(
    [ari_results["KMeans"], ari_results["EKM"]], 
    labels=["KMeans", "EKM"], patch_artist=True,
    boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red")
)
axs[0].set_title("Adjusted Rand Index (ARI) Distribution")
axs[0].set_ylabel("ARI Score")
# Silhouette
axs[1].boxplot(
    [sil_results["KMeans"], sil_results["EKM"]], 
    labels=["KMeans", "EKM"], patch_artist=True,
    boxprops=dict(facecolor="lightgreen"), medianprops=dict(color="red")
)
axs[1].set_title("Silhouette Distribution")
axs[1].set_ylabel("Silhouette Score")
plt.suptitle("Monte Carlo Benchmark ({} runs)".format(N_REPEATS), fontsize=14)
plt.show()