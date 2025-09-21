# API 参考（中文）

详细手写文档：
- 英文：[`ekm_api.md`](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/blob/main/python/ekm_api.md)
- 中文：[`ekm_api_zh.md`](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/blob/main/python/README_zh.md)

## 直接跳转
- [EKM 全量批](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/blob/main/python/ekm_api.md)
- [MiniBatchEKM](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/blob/main/python/ekm_api_zh.md)

## 说明
当前使用手写 Markdown 以便更好控制公式与中英文双语。`metric='euclidean'` 下初始化已改为使用 sklearn `kmeans_plusplus`；其它距离继续内部回退。将来若拆分为包，可启用 `mkdocstrings` 自动提取。
