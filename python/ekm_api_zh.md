# EKM 与 MiniBatchEKM API 参考（中文）

> 对 `ekm_sklearn.py` 中实现的全量批 `EKM` 与小批量 `MiniBatchEKM` 的参数、属性、方法、数值稳定策略进行中文说明。英文版请见 `ekm_api.md`。

[English API](./ekm_api.md) | 当前为中文版

---
## 目录
- [EKM（全量批）](#ekm全量批)
  - [构造参数](#构造参数)
  - [fit 后属性](#fit-后属性)
  - [方法说明](#方法说明)
  - [行为特性与边界情况](#行为特性与边界情况)
  - [示例](#示例)
- [MiniBatchEKM](#minibatchekm)
  - [构造参数](#minibatchekm-构造参数)
  - [fit 后属性](#minibatchekm-fit-后属性)
  - [方法说明](#minibatchekm-方法说明)
  - [行为特性与边界情况](#minibatchekm-行为特性与边界情况)
  - [示例](#minibatchekm-示例)
- [公共工具函数](#公共工具函数)
- [数值稳定策略](#数值稳定策略)
- [版本信息](#版本信息)

---
## EKM（全量批）
Equilibrium K-Means 在标准 K-Means 的软分配基础上引入平衡校正项，缓解不平衡簇偏置。

### 构造参数
| 名称 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `n_clusters` | int | 3 | 聚类数 K。 |
| `metric` | str | `'euclidean'` | 距离度量，可选 `'euclidean'` / `'manhattan'`。|
| `alpha` | float 或 `'dvariance'` | 0.5 | 平滑参数；若为 `'dvariance'` 则按数据方差自动估计。|
| `scale` | float | 2.0 | 与 `'dvariance'` 结合使用的放缩系数。|
| `max_iter` | int | 500 | 单次运行最大迭代次数。|
| `tol` | float | 1e-3 | 相对中心变化停止阈值。|
| `n_init` | int | 1 | 多重随机重启次数；保留最好目标。|
| `init` | `'plus'` 或 ndarray | `'plus'` | 欧式距离下调用 sklearn `kmeans_plusplus`；其它距离使用内部回退；也可自定义中心。|
| `random_state` | int/None | None | 随机种子。|
| `use_numba` | bool | False | 是否启用 numba 加速权重计算。|
| `numba_threads` | int/None | None | 指定 numba 线程数。|

### fit 后属性
| 属性 | 形状 | 说明 |
|------|------|------|
| `cluster_centers_` | (K, d) | 最佳运行得到的中心。|
| `labels_` | (n,) | 硬标签。|
| `n_iter_` | int | 最佳运行迭代次数。|
| `objective_` | float | 近似目标值（\(\sum_i J_i\)）。|
| `alpha_` | float | 实际使用的 alpha。|
| `D_` | (n, K) | 最终距离矩阵（非平方）。|
| `W_` | (n, K) | 平衡权重。|
| `U_` | (n, K) | 归一化软分配。|

### 方法说明
| 方法 | 描述 |
|------|------|
| `fit(X, y=None)` | 运行（含 n_init 重启）并缓存结果。|
| `predict(X)` | 预测硬标签。|
| `fit_predict(X, y=None)` | 拟合并返回标签。|
| `transform(X)` | 返回到当前中心的距离矩阵。|
| `fit_transform(X, y=None)` | 拟合后返回 `D_`。|
| `membership(X)` | 返回稳定软分配矩阵。|
| `fit_membership(X, y=None)` | 拟合后返回 `U_`。|

### 行为特性与边界情况
- 若某行权重全 0，会回退到最近中心的硬指派。
- `alpha` 过大可能导致数值集中，需适当减小 `scale`。 
- `metric='manhattan'` 时使用均值而非几何中位数，属近似。

### 示例
```python
from ekm_sklearn import EKM
import numpy as np
X = np.random.randn(1200, 6)
model = EKM(n_clusters=4, alpha='dvariance', scale=2.0, n_init=3, use_numba=True, random_state=0)
model.fit(X)
proba = model.membership(X[:10])
```

---
## MiniBatchEKM
支持两种更新：累积（更接近全量结果）与在线（学习率加权）。

### MiniBatchEKM 构造参数
| 名称 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `n_clusters` | int | 3 | 聚类数。|
| `metric` | str | `'euclidean'` | 距离度量。|
| `alpha` | float 或 `'dvariance'` | 0.5 | 若 `'dvariance'` 则根据预热集估计。|
| `scale` | float | 2.0 | 自动 alpha 放缩。|
| `batch_size` | int | 256 | 小批量大小。|
| `max_epochs` | int | 10 | 最大 epoch 数。|
| `init` | `'plus'` 或 ndarray | `'plus'` | 初始中心策略。|
| `init_size` | int/None | None | 预热子集大小。|
| `shuffle` | bool | True | 每 epoch 是否洗牌。|
| `learning_rate` | float/None | None | None=累积；否则在线更新。|
| `tol` | float | 1e-3 | epoch 级收敛判据。|
| `reassignment_ratio` | float | 0.0 | 低权重跳过/重启阈值（相对 batch）。|
| `reassign_patience` | int | 3 | 连续多少次低权重后重启。|
| `verbose` | int | 1 | 日志级别。|
| `monitor_size` | int/None | 1024 | 近似目标监控子集。|
| `print_every` | int | 1 | 日志打印间隔（epoch）。|
| `use_numba` | bool | False | 是否启用 numba。|
| `numba_threads` | int/None | None | numba 线程数。|
| `random_state` | int/None | None | RNG 种子。|

### MiniBatchEKM fit 后属性
| 属性 | 形状 | 说明 |
|------|------|------|
| `cluster_centers_` | (K,d) | 训练后中心。|
| `alpha_` | float | 实际 alpha。|
| `n_epochs_` | int | 实际运行 epoch。|
| `counts_` | (K,) | 累积模式下的权重和。|
| `sums_` | (K,d) | 累积模式下的加权特征和。|
| `D_` | (n,K) | 最终距离。|
| `W_` | (n,K) | 平衡权重。|
| `U_` | (n,K) | 软分配。|
| `objective_approx_` | list | 每个 epoch 的近似目标轨迹。|

### MiniBatchEKM 方法说明
| 方法 | 描述 |
|------|------|
| `fit(X, y=None)` | 多 epoch 训练。|
| `partial_fit(X_batch)` | 流式/增量更新。|
| `predict(X)` | 硬标签。|
| `transform(X)` | 距离矩阵。|
| `membership(X)` | 软分配。|
| `fit_predict(X)` | 训练并返回标签。|
| `fit_membership(X)` | 训练并返回 `U_`。|

### MiniBatchEKM 行为特性与边界情况
- 累积模式对批次顺序敏感性较低；在线模式需谨慎选择 `learning_rate`。
- 重启仅在达到耐心阈值后触发（在线）。
- `monitor_size=None` 使用全数据监控（更慢但更精确）。

### MiniBatchEKM 示例
```python
from ekm_sklearn import MiniBatchEKM
import numpy as np
X = np.random.randn(15000, 10)
mb = MiniBatchEKM(n_clusters=5, alpha='dvariance', scale=2.0, batch_size=512, max_epochs=15)
mb.fit(X)
print(mb.objective_approx_[-3:])
```

---
## 公共工具函数
- `_pairwise_distance(X, Y=None, metric)`: 计算成对距离（未平方）。
- `_kmeans_plus_init(X, K, metric)`: k-means++ 初始中心（欧式=sklearn，非欧式=fallback）。
- `calc_weight(D2, alpha)`: 给定平方距离与 alpha 计算平衡权重（假定外部已做数值稳定处理）。

## 数值稳定策略
- 行内最小值移位后再 `exp`，降低下溢风险。
- 除法分母使用 `eps` 保护。
- 权重全 0 的行硬指派到最近中心。

## 版本信息
- 初始公开实现：2025-09。

---
如需更多实验性说明，可扩展本文件或查阅英文版文档。
