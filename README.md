# KESurv

[![PyPI](https://img.shields.io/pypi/v/kesurv.svg)](https://pypi.org/project/kesurv/)
[![Tests](https://github.com/yuvrajiro/kesurv/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvrajiro/kesurv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-kesurv.statml.in-0f6b66)](https://kesurv.statml.in)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yuvrajiro/kesurv/blob/main/LICENSE)

KESurv estimates patient-specific survival curves from right-censored data. A survival forest supplies adaptive neighbor weights; user-selected survival models vote on which neighbors have similar predicted survival trajectories; a weighted Kaplan–Meier estimator produces the final curve.

**Documentation:** <https://kesurv.statml.in>

## Install

```bash
pip install kesurv
pip install "kesurv[speed]"   # optional Numba acceleration
```

## Quick start

```python
from kesurv import KESurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree

model = KESurv(
    base_estimators=[
        ("ridge", CoxPHSurvivalAnalysis(alpha=1.0)),
        ("tree", SurvivalTree(max_depth=4, random_state=1)),
    ],
    forest=RandomSurvivalForest(n_estimators=100, random_state=1),
    epsilon=0.05,
    min_agreement=1,
    n_timepoints=128,
    random_state=1,
)
model.fit(X_train, y_train)

model.predict_survival(X_test, times=[1, 3, 5])   # survival probabilities
model.predict_survival_function(X_test)           # step functions
model.predict_neighbor_info(X_test)               # forest weights, votes, fallbacks
```

`X` must be finite and numeric (encode categories and impute upstream). `y` is a structured `(event, time)` array, as built by `sksurv.util.Surv.from_arrays`. Any base learner with `fit(X, y)` and `predict_survival_function(X)` works; see [the custom learner template](https://github.com/yuvrajiro/kesurv/blob/main/examples/custom_estimator.py).

## Features

- **Any base learners.** Use scikit-survival estimators or your own class.
- **Inspectable.** `predict_weights`, `predict_neighbor_info` and `kneighbors` show exactly which training subjects shaped each curve.
- **Configurable similarity grid.** Use `similarity_time`, `n_timepoints` or an explicit `similarity_times` grid.
- **Fast.** Training leaves, bootstrap multiplicities and curves are cached at fit time. Queries run in batches over forest-supported candidates only. Numba kernels are optional.
- **Extra outputs.** RMST (`predict_rmst`), risk scores (`predict`) and Harrell's C (`score`).

The weighted counting kernel builds on [fastkme](https://github.com/yuvrajiro/fastkme).

## Development

```bash
pip install -e ".[test,docs]"
python -m pytest -q
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

## Citation

If you use KESurv, please cite *KESurv: A Kernel Ensemble Method for Patient-Specific Survival Prediction*, Proceedings of Machine Learning Research (ACML 2026).

## License

MIT
