"""Evaluate survival learners and compute agreement on a common time grid."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # optional speed extra
    njit = None


def _evaluate_one(curve, times):
    if not callable(curve):
        raise TypeError("survival predictions must be callables or a numeric matrix")
    if hasattr(curve, "x") and hasattr(curve, "y"):
        knots = np.asarray(curve.x, dtype=np.float64)
        levels = np.asarray(curve.y, dtype=np.float64)
        if (knots.ndim != 1 or len(knots) == 0 or levels.shape != knots.shape
                or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0)):
            raise ValueError("a predicted step function has no valid time knots")
        positions = np.searchsorted(knots, times, side="right") - 1
        values = np.ones(len(times), dtype=np.float64)
        mask = positions >= 0
        values[mask] = (
            float(getattr(curve, "a", 1.0)) * levels[positions[mask]]
            + float(getattr(curve, "b", 0.0))
        )
        return values
    result = np.asarray(curve(times), dtype=np.float64)
    if result.shape == ():
        result = np.full(len(times), float(result))
    return result


def evaluate_curves(estimator, X, times, name):
    predictions = estimator.predict_survival_function(X)
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2 and np.issubdtype(
        predictions.dtype, np.number
    ):
        source_times = getattr(estimator, "unique_times_", None)
        if source_times is None:
            source_times = getattr(estimator, "event_times_", None)
        if source_times is None:
            raise ValueError(f"{name} returned a matrix but has no unique_times_ or event_times_")
        source_times = np.asarray(source_times, dtype=np.float64)
        if (predictions.shape != (len(X), len(source_times)) or len(source_times) == 0
                or not np.isfinite(source_times).all()
                or np.any(np.diff(source_times) <= 0)):
            raise ValueError(f"{name} returned a survival matrix with an invalid shape")
        positions = np.searchsorted(source_times, times, side="right") - 1
        values = np.ones((len(X), len(times)), dtype=np.float64)
        active = positions >= 0
        values[:, active] = predictions[:, positions[active]]
    else:
        if len(predictions) != len(X):
            raise ValueError(f"{name} returned {len(predictions)} curves for {len(X)} subjects")
        values = np.stack([_evaluate_one(curve, times) for curve in predictions])
    if values.shape != (len(X), len(times)) or not np.isfinite(values).all():
        raise ValueError(f"{name} produced invalid or nonfinite survival probabilities")
    if np.any(values < -1e-8) or np.any(values > 1 + 1e-8):
        raise ValueError(f"{name} produced probabilities outside [0, 1]")
    if np.any(np.diff(values, axis=1) > 1e-8):
        raise ValueError(f"{name} produced an increasing survival curve")
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def agreement_reference(train_curves, query_curves, candidate_indices, widths, cutoff):
    """Return exact agreement counts; useful as a correctness reference."""
    counts = np.zeros(len(candidate_indices), dtype=np.int32)
    if np.isinf(cutoff):
        counts[:] = train_curves.shape[0]
        return counts
    for learner in range(train_curves.shape[0]):
        area = np.abs(
            train_curves[learner, candidate_indices, :-1] - query_curves[learner, :-1]
        ).astype(np.float64) @ widths
        counts += area <= cutoff + 1e-12
    return counts


if njit is not None:

    @njit(cache=False)
    def _agreement_numba(train_curves, query_curves, indices, widths, cutoff, full_counts, minimum):
        counts = np.zeros(len(indices), dtype=np.int32)
        n_models = train_curves.shape[0]
        if np.isinf(cutoff):
            counts[:] = n_models
            return counts
        for k in range(len(indices)):
            count = 0
            for model in range(n_models):
                area = 0.0
                for point in range(len(widths)):
                    area += abs(
                        float(train_curves[model, indices[k], point])
                        - float(query_curves[model, point])
                    ) * widths[point]
                    if area > cutoff + 1e-12:
                        break
                if area <= cutoff + 1e-12:
                    count += 1
                if not full_counts and (count >= minimum or count + n_models - model - 1 < minimum):
                    break
            counts[k] = count
        return counts


def agreement_counts(train_curves, query_curves, candidate_indices, widths, cutoff,
                     minimum, backend="auto", full_counts=False):
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'")
    if backend == "numba" and njit is None:
        raise ImportError("Numba is not installed; install kesurv[speed]")
    if backend == "numpy" or (backend == "auto" and njit is None):
        return agreement_reference(train_curves, query_curves, candidate_indices, widths, cutoff)
    return _agreement_numba(train_curves, query_curves, candidate_indices, widths,
                            cutoff, full_counts, minimum)

