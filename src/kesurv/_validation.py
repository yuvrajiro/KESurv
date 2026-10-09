"""Input and estimator validation."""

from __future__ import annotations

import numpy as np


def validate_x(X, *, n_features=None, feature_names=None):
    names = None
    if hasattr(X, "columns"):
        names = np.asarray(X.columns, dtype=object)
        if any(not isinstance(name, str) for name in names):
            names = None
        if feature_names is not None and not np.array_equal(names, feature_names):
            raise ValueError("DataFrame columns must match the columns used at fit time")
    elif feature_names is not None:
        # Arrays remain valid after fitting on a DataFrame, as in scikit-learn.
        names = None
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional numeric array")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values; preprocess missing values first")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features; expected {n_features}")
    return X, names


def validate_y(y, n_samples):
    y = np.asarray(y)
    if y.dtype.names is None or len(y.dtype.names) != 2:
        raise ValueError("y must be a structured array with (event, time) fields")
    if len(y) != n_samples:
        raise ValueError("X and y must have the same number of samples")
    event = np.asarray(y[y.dtype.names[0]])
    time = np.asarray(y[y.dtype.names[1]], dtype=np.float64)
    if event.dtype.kind != "b":
        raise ValueError("the first y field must contain boolean event indicators")
    if not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("observed times must be finite and nonnegative")
    if not event.any():
        raise ValueError("at least one observed event is required")
    if time.max() <= 0:
        raise ValueError("maximum follow-up time must be positive")
    return event, time


def validate_base_estimators(estimators, min_agreement):
    if not estimators:
        raise ValueError("base_estimators must contain at least one named estimator")
    names = []
    for item in estimators:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("base_estimators must be (name, estimator) pairs")
        name, estimator = item
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("base estimator names must be unique nonempty strings")
        if not callable(getattr(estimator, "fit", None)) or not callable(
            getattr(estimator, "predict_survival_function", None)
        ):
            raise TypeError(
                f"base estimator {name!r} must implement fit(X, y) "
                "and predict_survival_function(X)"
            )
        names.append(name)
    if isinstance(min_agreement, (bool, np.bool_)) or not isinstance(
        min_agreement, (int, np.integer)
    ) or not 1 <= min_agreement <= len(names):
        raise ValueError("min_agreement must be an integer from 1 to len(base_estimators)")


def validate_grid(similarity_times, similarity_time, n_timepoints, max_time):
    if similarity_times is not None:
        if similarity_time is not None or n_timepoints != 128:
            raise ValueError(
                "similarity_times cannot be combined with similarity_time or "
                "a nondefault n_timepoints"
            )
        grid = np.asarray(similarity_times, dtype=np.float64)
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("similarity_times must be a one-dimensional grid of at least two points")
        if not np.isfinite(grid).all() or grid[0] != 0 or np.any(np.diff(grid) <= 0):
            raise ValueError("similarity_times must start at 0 and increase strictly")
    else:
        if isinstance(n_timepoints, (bool, np.bool_)) or not isinstance(
            n_timepoints, (int, np.integer)
        ) or n_timepoints < 2:
            raise ValueError("n_timepoints must be an integer of at least 2")
        horizon = max_time if similarity_time is None else float(similarity_time)
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("similarity_time must be finite and positive")
        grid = np.linspace(0.0, horizon, n_timepoints)
    return grid


def validate_times(times):
    arr = np.asarray(times, dtype=np.float64)
    if arr.ndim > 1 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("prediction times must be finite and nonnegative")
    return np.atleast_1d(arr)
