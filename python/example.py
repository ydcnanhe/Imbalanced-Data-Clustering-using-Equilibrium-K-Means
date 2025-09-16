import numpy as np
from ekm_sklearn import EKM
# 假设 X 是 (N, P) 矩阵
X = np.random.rand(200, 2)
C_init=np.array([[1,0],[0,1]], dtype=np.float64)
model = EKM(n_clusters=2, metric='euclidean', alpha=0.5, init=C_init)
labels = model.fit_predict(X)
print("聚类标签:", labels[:20])
print("隶属度矩阵",model.U_[:20])
print("簇中心:\n", model.cluster_centers_)
print("迭代次数:", model.n_iter_)
print("目标函数值 J:", model.objective_)
# predict 新数据
X_new = np.random.rand(5, 2)
print("新数据预测簇:", model.predict(X_new))
# membership 得到隶属度矩阵
print("隶属度矩阵:\n", model.membership(X_new))
# transform 得到距离矩阵
print("距离矩阵:\n", model.transform(X_new))