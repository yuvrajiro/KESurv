# FAQ and limits

## Can I supply my own forest?

Yes. Pass `forest=` with an unfitted compatible estimator that exposes `fit`, `apply`, and `estimators_samples_`. KESurv fits a copy. The default is a 100-tree `RandomSurvivalForest`.

## Does `similarity_time` cap predicted survival time?

No. It sets the comparison horizon for base-model agreement. The weighted Kaplan–Meier output follows training event times. Use `predict_survival(X, times)` for requested evaluation times.

## What if no base model agrees?

KESurv falls back to the original forest weights among its candidates. Inspect `predict_neighbor_info(X)` to identify such queries.

## Can I pass a pandas DataFrame?

Yes, when every feature is finite and numeric. KESurv records feature names and checks their order at prediction time. Encode categorical columns and impute missing values first.

## Is this a drop-in medical decision tool?

No clinical validity is implied by the package. Evaluate calibration, discrimination, censoring assumptions, subgroup behavior, and the base models on data appropriate for the intended use.

## Why might I see a delay on the first prediction?

The optional Numba backend compiles kernels on first use. Subsequent calls avoid that cost. Set `backend="numpy"` for a no-JIT path.
