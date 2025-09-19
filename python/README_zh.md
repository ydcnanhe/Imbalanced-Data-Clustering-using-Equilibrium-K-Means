# 不平衡数据聚类：Equilibrium K-Means (EKM)

> 本仓库提供 Equilibrium K-Means (EKM) 及其 Mini-Batch 版本的研究实现，适用于强不平衡数据集的聚类。包含若干基准脚本、参数敏感性实验工具、以及可选的 numba 加速。

[English README](./README.md) | 当前为中文版 | API: [中文](./ekm_api_zh.md) | [English](./ekm_api.md)

## 文件概览
| 文件 | 说明 |
|------|------|
| `ekm_sklearn.py` | 核心实现：全量批 `EKM` 与 `MiniBatchEKM`，含 `_pairwise_distance`、k-means++ 初始化与权重计算。支持 `use_numba=True`。|
| `example.py` | 快速示例：演示全量批 EKM 以及 MiniBatchEKM 累积模式。|
| `benchmark.py` | Monte Carlo：`KMeans` vs `EKM`（ARI 与 Silhouette）。|
| `benchmark_alphaSweep.py` | 扫描 `scale`（影响自动 α）的敏感性。|
| `benchmark_minibatch_compare.py` | 对比 Full Batch vs MiniBatch（累积 / 在线）性能（时间、ARI、NMI、objective）。|
| `benchmark_numba_ekm.py` | 对比 EKM 在开启/关闭 numba 下的耗时与加速比。|
| `ekm_api.md` / `ekm_api_zh.md` | EKM 与 MiniBatchEKM API 文档（中英文）。|
| `requirements.txt` | 运行依赖；可选安装 `numba`、`matplotlib`、以及文档依赖。|

## 算法概要
Equilibrium K-Means 通过引入平衡加权（equilibrium weight）缓解经典 K-Means 在不平衡样本上的偏置：
1. 计算样本到中心的平方距离 $$D_{ik}^2$$。
2. 稳定的软分配：$$U_{ik} \propto e^{-\alpha D_{ik}^2}$$（对每行减去最小值防止下溢）。
3. 平衡权重：$$W_{ik} = U_{ik} (1 - \alpha (D_{ik}^2 - J_i))$$，其中 $$J_i = \sum_k U_{ik} D_{ik}^2$$。
4. 用 $$W$$ 做加权更新中心；若整行权重为 0，回退到最近中心的硬指派。
5. 监控目标近似：$$\sum_i J_i$$。

Mini-Batch 版本：
- 累积模式（默认）：维护全局加权和与权重和，收敛行为接近全量批。
- 在线模式：固定学习率对批内加权均值做指数平滑，收敛更快但依赖超参。

额外特性：行移位防下溢、空簇耐心重启、epoch 近似目标监控、numba 加速。

## 安装
在 `python` 目录：
```bash
pip install -r requirements.txt
# 可选：加速与绘图
pip install numba matplotlib
# 可选：文档构建
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
```
Windows 建议使用：`py -m pip install ...`。

## 快速开始
```python
import numpy as np
from ekm_sklearn import EKM
X = np.random.randn(500, 4)
model = EKM(n_clusters=3, alpha='dvariance', scale=2.0, max_iter=300, tol=1e-3, random_state=0)
model.fit(X)
print(model.cluster_centers_.shape, model.objective_)
```
Mini-Batch：
```python
from ekm_sklearn import MiniBatchEKM
mb = MiniBatchEKM(n_clusters=3, alpha='dvariance', scale=2.0, batch_size=256, max_epochs=10)
mb.fit(X)
```
流式：
```python
stream = MiniBatchEKM(n_clusters=3, alpha='dvariance', scale=2.0, batch_size=128)
for chunk in np.array_split(X, 10):
    stream.partial_fit(chunk)
```

## numba 加速
```python
model = EKM(n_clusters=3, alpha='dvariance', scale=2.0, use_numba=True)
model.fit(X)
```
首轮会触发 JIT 编译，建议预热小批。

## 基准脚本
1. `benchmark.py`: KMeans vs EKM (ARI / Silhouette)。
2. `benchmark_alphaSweep.py`: `scale` 敏感性。
3. `benchmark_minibatch_compare.py`: 全量 vs 累积 vs 在线。
4. `benchmark_numba_ekm.py`: numba 加速对比。

运行示例：
```bash
python benchmark_minibatch_compare.py
python benchmark_numba_ekm.py
```

## 推荐超参
| 场景 | 建议 |
|------|------|
| 强不平衡中等规模 | `alpha='dvariance', scale=2.0`, `tol=1e-3` |
| 大规模内存压力 | MiniBatch 累积 + `batch_size` 256–1024 |
| 流式快速适应 | 在线模式 `learning_rate` 0.05–0.3 & `reassign_patience>=3` |
| 需要更平滑 | 降低 `scale` 或减小 `alpha` |
| 性能测试 | 启用 `use_numba=True` 并预热 |

## 可复现性
- 固定 `random_state`。
- numba 并行浮点归约可能带来细微差异，可用目标值近似比较。

## API 文档
- 英文：`ekm_api.md`
- 中文：`ekm_api_zh.md`

## 构建站点文档 (MkDocs)
```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
mkdocs serve  # 本地预览
mkdocs build  # 生成静态站点 (site/)
```

## 引用
请引用：
- He Yudong, *An Equilibrium Approach to Clustering: Surpassing Fuzzy C-Means on Imbalanced Data*, IEEE TFS, 2025.
- He Yudong, *Imbalanced Data Clustering Using Equilibrium K-Means*, arXiv, 2024.

## 许可证
GNU GPL v3。

## 后续计划
- 单元测试与 CI
- 学习率自适应
- Pybind11 / C++ 后端
- 更稳定的 L1 中心更新策略

---
问题或建议欢迎反馈。
