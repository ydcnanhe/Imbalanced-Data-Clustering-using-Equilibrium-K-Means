# API Reference (English)

The detailed API is authored manually in Markdown to keep mathematical notation clean.

- Core file: `ekm_sklearn.py` (Euclidean k-means++ seeding now delegated to sklearn's implementation)
- Full-batch: see [`ekm_api.md`](../ekm_api.md)
- Mini-batch: also in [`ekm_api.md`](../ekm_api.md)
- Chinese version: [`ekm_api_zh.md`](../ekm_api_zh.md)

## Direct Links
- [EKM (Full Batch)](../ekm_api.md#ekm全量批) *(Chinese anchor if viewed there)*
- [MiniBatchEKM](../ekm_api.md#minibatchekm)

## Note on mkdocstrings
Currently the project exposes classes in a single file. If later reorganized into a package, we can switch to automatic extraction:
```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: false
```

## Stability
Weight computation uses a per-row shift before exponentials to avoid underflow.
