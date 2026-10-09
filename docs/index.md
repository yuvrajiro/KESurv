# KESurv

<p class="ks-lead">Patient-specific survival curves from right-censored data. A survival forest proposes neighbors, your own survival models vote on which neighbors look alike, and a weighted Kaplan–Meier estimator turns the selected neighbors into a curve.</p>

<div class="ks-links">
  <a href="getting-started.html">Get started →</a>
  <a href="concepts.html">How it works</a>
  <a href="api.html">API reference</a>
  <a href="https://github.com/yuvrajiro/kesurv">GitHub</a>
</div>

## Install

```console
pip install kesurv
```

Add the optional Numba kernels with `pip install "kesurv[speed]"`. Python 3.10 or newer is required.

## Example

```python
from kesurv import KESurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree

model = KESurv(
    base_estimators=[
        ("cox", CoxPHSurvivalAnalysis(alpha=1.0)),
        ("tree", SurvivalTree(max_depth=4, random_state=7)),
    ],
    forest=RandomSurvivalForest(n_estimators=100, random_state=7),
    epsilon=0.05,
    min_agreement=1,
    random_state=7,
).fit(X_train, y_train)

model.predict_survival(X_test, times=[1.0, 2.0, 3.0])  # (n_test, 3) probabilities
model.predict_neighbor_info(X_test[:1])                 # which neighbors, and why
```

`X` is a finite numeric matrix and `y` a scikit-survival structured array (`Surv.from_arrays`). [Getting started](getting-started.md) has a complete, runnable version.

## The method in one line

<div class="ks-flow">survival forest → candidate neighbors → base-model agreement → weighted Kaplan–Meier</div>

- **Bring any learner.** Any object with `fit` and `predict_survival_function` can vote, including scikit-survival models and [your own](custom-estimators.md).
- **Inspect every prediction.** Forest weights, agreement counts, selected neighbors and fallbacks are all exposed ([predictions](predictions.md)).
- **Control the cost.** The similarity horizon, grid resolution, batch size and NumPy or Numba backend are all configurable ([performance](performance.md)).

![Three KESurv patient-specific survival curves compared with a population Kaplan–Meier curve on synthetic data.](_static/synthetic-survival-curves.png)

The figure comes from the [synthetic notebook](synthetic-demo.ipynb).

## Citation

KESurv implements the method in *KESurv: A Kernel Ensemble Method for Patient-Specific Survival Prediction*, Proceedings of Machine Learning Research (ACML 2026).

```{toctree}
:hidden:

getting-started
concepts
predictions
custom-estimators
synthetic-demo
performance
api
faq
contributing
```
