# Getting started

KESurv expects a numeric feature matrix and scikit-survival's structured outcome array. This example creates a small reproducible dataset, fits a forest and two complementary survival trees, then asks for survival probabilities.

## Install

```console
pip install kesurv
pip install "kesurv[speed]"   # optional Numba kernels
```

Python 3.10 or newer is required. The `speed` extra enables JIT kernels; the NumPy implementation is always available.

## Fit and predict

```python
import numpy as np
from kesurv import KESurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from sksurv.util import Surv

rng = np.random.default_rng(7)
X = rng.normal(size=(100, 5))
event_time = rng.exponential(scale=np.exp(-0.3 * X[:, 0] + 0.2 * X[:, 1]))
censor_time = rng.exponential(scale=2.0, size=len(X))
observed_time = np.minimum(event_time, censor_time)
y = Surv.from_arrays(event_time <= censor_time, observed_time)

model = KESurv(
    base_estimators=[
        ("shallow", SurvivalTree(max_depth=3, random_state=1)),
        ("deep", SurvivalTree(max_depth=5, random_state=2)),
    ],
    forest=RandomSurvivalForest(n_estimators=40, random_state=3),
    epsilon=0.05,
    min_agreement=1,
    similarity_time=2.0,
    n_timepoints=64,
    random_state=7,
)
model.fit(X[:80], y[:80])

survival = model.predict_survival(X[80:83], times=[0.5, 1.0, 2.0])
print(survival.shape)  # (3, 3)

curves = model.predict_survival_function(X[80:83])
print(curves[0](1.0))
```

The horizon `similarity_time` affects **which neighbors agree**; it does not truncate the final survival curve. The final curve is defined at the distinct training event times.

## Input checklist

- `X` has one row per subject and only finite numeric values. Encode categorical features and impute missing values upstream, for example with a scikit-learn pipeline.
- `y` is a two-field structured array: boolean event status, then nonnegative observed time. `sksurv.util.Surv.from_arrays` builds it.
- Each base model must implement `fit(X, y)` and `predict_survival_function(X)`.
- Each prediction query uses the same feature columns and order as training.

For your own learner, follow the [custom estimator guide](custom-estimators.md). For all return types, see [Predictions and inspection](predictions.md).
