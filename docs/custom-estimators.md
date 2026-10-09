# Bring your own survival estimator

`base_estimators` is a sequence of `(name, estimator)` pairs. Each estimator needs two methods: `fit(X, y)` and `predict_survival_function(X)`. KESurv copies the supplied estimator before fitting; scikit-learn compatible objects are cloned, and simpler objects are deep copied.

The easiest path is a scikit-survival estimator:

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree

base_estimators = [
    ("cox", CoxPHSurvivalAnalysis(alpha=1.0)),
    ("tree", SurvivalTree(max_depth=4, random_state=11)),
]
```

Choose learners with genuinely different inductive biases when possible; KESurv treats each vote equally.

## Minimal template

```python
class MySurvivalEstimator:
    def fit(self, X, y):
        # Learn from numeric X and a structured (event, time) array.
        # Store any fitted state on self.
        return self

    def predict_survival_function(self, X):
        # Return one callable survival function for each row in X.
        # Each curve receives a vector of times and returns probabilities.
        return [self._curve_for_row(row) for row in X]
```

The repository includes a fully working population Kaplan–Meier example in `examples/custom_estimator.py`. You may also return a numeric 2D matrix if the estimator exposes strictly increasing `unique_times_` or `event_times_` aligned with its columns. Curves must be finite, lie in `[0, 1]`, and never increase. A stable callable should return survival probability one before its first event time.

For cross-validation and parameter search, subclass `sklearn.base.BaseEstimator` so scikit-learn can clone your class and expose its parameters. A plain Python class can work if `deepcopy` succeeds.

## Preprocessing belongs with the learner

KESurv expects numeric, finite `X`. Encode categories and impute missing values before fitting. If a custom learner needs its own preprocessing, put it inside the learner's `fit` and prediction methods so training and query transformations remain identical. The forest still receives the shared numeric input matrix.
