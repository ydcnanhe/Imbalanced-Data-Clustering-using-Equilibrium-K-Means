# Equilibrium K-Means (EKM)

Welcome to the documentation site.

## Overview
This site provides:
- Conceptual summary of Equilibrium K-Means (EKM)
- Mini-batch variant (accumulation / online)
- Numerically stable soft membership weighting
- Benchmarks and recommended hyperparameters

For full README details see the repository root `README.md`.

## Quick Usage
```python
from ekm_sklearn import EKM
model = EKM(n_clusters=3, alpha='dvariance', scale=2.0)
model.fit(X)
```

## API
See the API page or the Markdown sources:
- English: `ekm_api.md`
- Chinese: `ekm_api_zh.md`

## License
GPL v3.
