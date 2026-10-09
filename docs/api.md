# API reference

```{autoclass} kesurv.KESurv
:members: fit, predict_survival_function, predict_survival, predict_survival_at, predict_rmst, predict, score, predict_weights, predict_neighbor_info, kneighbors
:member-order: bysource
```

## Fitted attributes

| Attribute | Meaning |
| --- | --- |
| `forest_` | Fitted copy of the survival forest |
| `base_estimators_` | Fitted `(name, estimator)` pairs |
| `unique_times_` | Distinct training event times; columns of `return_array=True` |
| `similarity_times_` | Grid used for learner agreement |
| `n_features_in_` | Feature count seen during fit |
| `feature_names_in_` | Input feature names, when supplied by the input |
| `n_training_samples_` | Number of training subjects |
