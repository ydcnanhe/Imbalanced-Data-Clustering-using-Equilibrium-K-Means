# Equilibrium K-Means (EKM) 文档

欢迎访问文档站点。

## 简介
本页面提供：
- Equilibrium K-Means (EKM) 核心思想
- Mini-Batch 版本（累积 / 在线）
- 数值稳定的软分配与平衡权重
- 基准测试与推荐超参数

完整细节请参阅仓库根目录 `README_zh.md`。

## 快速示例
```python
from ekm_sklearn import EKM
model = EKM(n_clusters=3, alpha='dvariance', scale=2.0)
model.fit(X)
```

## API
- 英文：`ekm_api.md`
- 中文：`ekm_api_zh.md`

## 许可证
GPL v3。
