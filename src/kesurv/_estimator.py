"""Public KESurv estimator."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from ._curve_distance import agreement_counts, evaluate_curves
from ._forest_weights import ForestWeights
from ._step_function import SurvivalStepFunction
from ._validation import (
    validate_base_estimators,
    validate_grid,
    validate_times,
    validate_x,
    validate_y,
)
from ._weighted_km import WeightedKM


def _fresh_estimator(estimator):
    try:
        return clone(estimator)
    except (TypeError, ValueError):
        # A minimal custom learner only needs fit and predict_survival_function.
        # Deepcopy prevents the caller's instance from being fitted in place.
        try:
            return deepcopy(estimator)
        except Exception as exc:
            raise TypeError(
                "could not copy a custom base estimator; implement sklearn "
                "get_params/set_params or make it deepcopy-compatible"
            ) from exc


class KESurv(BaseEstimator):
    """Kernel ensemble survival estimator for right-censored outcomes.

    Parameters
    ----------
    base_estimators : sequence of (str, estimator)
        User-selected survival learners. Each learner must implement ``fit``
        and ``predict_survival_function``. Predictions can be a sequence of
        callable curves or a matrix aligned with ``unique_times_``.
    epsilon : float, default=0.05
        Maximum normalized absolute area for a learner's agreement vote.
    min_agreement : int, default=1
        Number of base learners that must agree.
    forest : RandomSurvivalForest-like estimator, optional
        Must expose ``fit``, ``apply``, and ``estimators_samples_``.
    similarity_time : float, optional
        Similarity integration horizon. Defaults to maximum training follow-up.
    n_timepoints : int, default=128
        Uniform similarity grid size, including zero and the horizon.
    similarity_times : array-like, optional
        Explicit strictly increasing similarity grid starting at zero.
    backend : {'auto', 'numpy', 'numba'}, default='auto'
        Kernel implementation. Numba is optional.
    batch_size : int, default=64
        Number of query subjects evaluated together by the base learners.

    Notes
    -----
    The weighted Kaplan-Meier calculation includes events recorded at time
    zero. This is consistent with S(t) = P(T > t), while the paper writes its
    product over strictly positive event times.
    """

    def __init__(
        self,
        base_estimators,
        epsilon=0.05,
        min_agreement=1,
        forest=None,
        similarity_time=None,
        n_timepoints=128,
        similarity_times=None,
        random_state=None,
        backend="auto",
        batch_size=64,
    ):
        self.base_estimators = base_estimators
        self.epsilon = epsilon
        self.min_agreement = min_agreement
        self.forest = forest
        self.similarity_time = similarity_time
        self.n_timepoints = n_timepoints
        self.similarity_times = similarity_times
        self.random_state = random_state
        self.backend = backend
        self.batch_size = batch_size

    def fit(self, X, y):
        X, names = validate_x(X)
        event, time = validate_y(y, len(X))
        validate_base_estimators(self.base_estimators, self.min_agreement)
        epsilon = float(self.epsilon)
        if np.isnan(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be nonnegative or infinity")
        if self.backend not in ("auto", "numpy", "numba"):
            raise ValueError("backend must be 'auto', 'numpy', or 'numba'")
        if isinstance(self.batch_size, (bool, np.bool_)) or not isinstance(
            self.batch_size, (int, np.integer)
        ) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        grid = validate_grid(
            self.similarity_times, self.similarity_time, self.n_timepoints, time.max()
        )
        forest = (
            RandomSurvivalForest(n_estimators=100, random_state=self.random_state)
            if self.forest is None else _fresh_estimator(self.forest)
        )
        forest.fit(X, y)
        forest_weights = ForestWeights(forest, X)
        fitted = []
        training_curves = []
        for name, estimator in self.base_estimators:
            learner = _fresh_estimator(estimator)
            learner.fit(X, y)
            training_curves.append(evaluate_curves(learner, X, grid, name))
            fitted.append((name, learner))
        self.forest_ = forest
        self.base_estimators_ = fitted
        self._forest_weights = forest_weights
        self._train_curves = np.ascontiguousarray(np.stack(training_curves))
        self._km = WeightedKM(event, time)
        self.unique_times_ = self._km.event_times
        self.similarity_times_ = grid
        self._widths = np.ascontiguousarray(np.diff(grid), dtype=np.float64)
        self._cutoff = epsilon * grid[-1]
        self.n_features_in_ = X.shape[1]
        if names is not None:
            self.feature_names_in_ = names
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_training_samples_ = len(X)
        self._event = event
        self._time = time
        return self

    def _validated_query(self, X):
        check_is_fitted(self, "base_estimators_")
        values, _ = validate_x(
            X,
            n_features=self.n_features_in_,
            feature_names=getattr(self, "feature_names_in_", None),
        )
        return values

    def _predict_core(self, X, *, survival, details):
        X = self._validated_query(X)
        for start in range(0, len(X), self.batch_size):
            batch = X[start : start + self.batch_size]
            leaves = self._forest_weights.apply(batch)
            query_curves = None
            if not np.isinf(self._cutoff):
                query_curves = np.ascontiguousarray(
                    np.stack([
                        evaluate_curves(estimator, batch, self.similarity_times_, name)
                        for name, estimator in self.base_estimators_
                    ])
                )
            for position in range(len(batch)):
                candidates, forest_weights = self._forest_weights.one(leaves[position])
                if query_curves is None:
                    counts = np.full(len(candidates), len(self.base_estimators_), dtype=np.int32)
                else:
                    counts = agreement_counts(
                        self._train_curves,
                        np.ascontiguousarray(query_curves[:, position, :]),
                        candidates,
                        self._widths,
                        self._cutoff,
                        self.min_agreement,
                        backend=self.backend,
                        full_counts=details,
                    )
                selected = candidates[counts >= self.min_agreement]
                mass = forest_weights[selected].sum()
                fallback = mass <= 0.0
                indices = candidates if fallback else selected
                selected_weights = forest_weights[indices].copy()
                selected_weights /= selected_weights.sum()
                result = None
                if survival:
                    result = self._km.predict(indices, selected_weights, backend=self.backend)
                info = None
                if details:
                    info = {
                        "candidate_indices": candidates,
                        "forest_weights": forest_weights[candidates],
                        "agreement_counts": counts,
                        "selected_indices": indices,
                        "selected_weights": selected_weights,
                        "fallback": fallback,
                    }
                yield result, indices, selected_weights, info

    def predict_survival_function(self, X, return_array=False):
        curves = np.stack([
            result for result, _, _, _ in self._predict_core(X, survival=True, details=False)
        ])
        if return_array:
            return curves
        if self.unique_times_[0] > 0:
            knots = np.concatenate(([0.0], self.unique_times_))
            values = np.column_stack((np.ones(len(curves)), curves))
        else:
            knots = self.unique_times_
            values = curves
        return np.asarray([SurvivalStepFunction(knots, row) for row in values], dtype=object)

    def predict_survival(self, X, times):
        times = validate_times(times)
        curves = self.predict_survival_function(X, return_array=True)
        positions = np.searchsorted(self.unique_times_, times, side="right") - 1
        result = np.ones((len(curves), len(times)), dtype=np.float64)
        active = positions >= 0
        result[:, active] = curves[:, positions[active]]
        return result

    def predict_survival_at(self, X, times):
        return self.predict_survival(X, times)

    def predict_weights(self, X):
        X = self._validated_query(X)
        matrix = np.zeros((len(X), self.n_training_samples_), dtype=np.float64)
        for row, (_, indices, weights, _) in enumerate(
            self._predict_core(X, survival=False, details=False)
        ):
            matrix[row, indices] = weights
        return matrix

    def predict_neighbor_info(self, X):
        return [info for _, _, _, info in self._predict_core(X, survival=False, details=True)]

    def kneighbors(self, X, n_neighbors=5):
        """Return the top training indices and their final KESurv weights."""
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        matrix = self.predict_weights(X)
        count = min(n_neighbors, matrix.shape[1])
        indices = np.argsort(-matrix, axis=1, kind="stable")[:, :count]
        weights = np.take_along_axis(matrix, indices, axis=1)
        return indices, weights

    def predict_rmst(self, X, horizon):
        """Restricted mean survival time through a finite nonnegative horizon."""
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError("horizon must be finite and nonnegative")
        X = self._validated_query(X)
        if horizon == 0:
            return np.zeros(len(X))
        knots = np.concatenate(([0.0], self.unique_times_[
            (self.unique_times_ > 0) & (self.unique_times_ < horizon)
        ], [horizon]))
        values = self.predict_survival(X, knots[:-1])
        return values @ np.diff(knots)

    def predict(self, X):
        """Return a risk score equal to negative RMST through max follow-up."""
        return -self.predict_rmst(X, float(self._time.max()))

    def score(self, X, y):
        """Harrell concordance of the negative-RMST risk score."""
        event, time = validate_y(y, len(X))
        return concordance_index_censored(event, time, self.predict(X))[0]
