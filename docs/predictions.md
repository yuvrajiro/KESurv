# Predictions and inspection

All prediction methods operate on a fitted `KESurv` instance and accept the same numeric feature layout used for fitting.

| Method | Returns | Typical use |
| --- | --- | --- |
| `predict_survival_function(X)` | Array of callable step functions | Plot a full curve for each subject |
| `predict_survival_function(X, return_array=True)` | Subjects × training event times | Vectorized curve analysis |
| `predict_survival(X, times)` | Subjects × requested times | Fixed-horizon survival probabilities |
| `predict_rmst(X, horizon)` | One RMST per subject | Restricted mean survival time |
| `predict(X)` | Negative RMST through maximum training follow-up | Risk ranking and scikit-learn scoring |
| `score(X, y)` | Harrell concordance | Basic held-out discrimination |
| `predict_weights(X)` | Subjects × training subjects | Full final weight matrix |
| `predict_neighbor_info(X)` | List of per-subject dictionaries | Audit candidates, votes, fallback |
| `kneighbors(X, n_neighbors)` | Indices and weights | Highest-weight training subjects |

`predict_survival_at(X, times)` is an alias of `predict_survival(X, times)`. Probabilities before the first training event are one; beyond the last event they hold the last observed survival level.

## Ask why a subject received a curve

```python
info = model.predict_neighbor_info(X_test[:1])[0]
print(info["candidate_indices"])
print(info["forest_weights"])
print(info["agreement_counts"])
print(info["selected_indices"])
print(info["selected_weights"])
print(info["fallback"])
```

`candidate_indices` and `forest_weights` show the forest-supported training subjects. `agreement_counts` counts base learners that voted for each candidate. `selected_indices` and `selected_weights` are the observations used by weighted Kaplan–Meier. When `fallback` is true, the agreement filter passed no positively weighted candidate and the original forest weights were used.

```python
indices, weights = model.kneighbors(X_test[:2], n_neighbors=5)
```

These are **weight rankings**, not distances in feature space. `predict_weights` allocates a dense matrix, so use neighbor information or `kneighbors` when you only need a small inspection result.

## Plot a curve

```python
import matplotlib.pyplot as plt

curve = model.predict_survival_function(X_test[:1])[0]
times = model.unique_times_
plt.step(times, curve(times), where="post")
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.ylim(0, 1.02)
plt.show()
```

Matplotlib is needed only for this plotting example.
