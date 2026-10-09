"""Weighted Kaplan-Meier with training-time sort and tie groups cached once.

The grouped counting approach follows Rahul Goswami's MIT-licensed fastkme:
https://github.com/yuvrajiro/fastkme
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _km_reference(group_id, event, event_group_ids, n_groups, indices, weights):
    total = np.bincount(group_id[indices], weights=weights, minlength=n_groups)
    deaths = np.bincount(group_id[indices[event[indices]]],
                         weights=weights[event[indices]], minlength=n_groups)
    risk = np.cumsum(total[::-1])[::-1]
    hazard = np.zeros(n_groups, dtype=np.float64)
    np.divide(deaths, risk, out=hazard, where=risk > 0)
    return np.cumprod(np.maximum(0.0, 1.0 - hazard))[event_group_ids]


if njit is not None:

    @njit(cache=False)
    def _km_numba(group_id, event, event_group_ids, n_groups, indices, weights):
        total = np.zeros(n_groups, dtype=np.float64)
        deaths = np.zeros(n_groups, dtype=np.float64)
        for k in range(len(indices)):
            index = indices[k]
            group = group_id[index]
            total[group] += weights[k]
            if event[index]:
                deaths[group] += weights[k]
        survival = np.empty(n_groups, dtype=np.float64)
        risk = 0.0
        for group in range(n_groups - 1, -1, -1):
            risk += total[group]
            survival[group] = risk
        running = 1.0
        for group in range(n_groups):
            if survival[group] > 0.0 and deaths[group] > 0.0:
                running *= max(0.0, 1.0 - deaths[group] / survival[group])
            survival[group] = running
        result = np.empty(len(event_group_ids), dtype=np.float64)
        for k in range(len(event_group_ids)):
            result[k] = survival[event_group_ids[k]]
        return result


class WeightedKM:
    def __init__(self, event, time):
        self.event = np.asarray(event, dtype=np.bool_)
        observed_times, self.group_id = np.unique(time, return_inverse=True)
        self.n_groups = len(observed_times)
        self.event_times = np.unique(time[event])
        self.event_group_ids = np.searchsorted(observed_times, self.event_times)

    def predict(self, indices, weights, backend="auto"):
        indices = np.asarray(indices, dtype=np.intp)
        weights = np.asarray(weights, dtype=np.float64)
        if len(indices) != len(weights) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("weighted KM requires matching indices and positive weight mass")
        if backend == "numba" and njit is None:
            raise ImportError("Numba is not installed; install kesurv[speed]")
        if backend in ("auto", "numba") and njit is not None:
            return _km_numba(self.group_id, self.event, self.event_group_ids,
                             self.n_groups, indices, weights)
        return _km_reference(self.group_id, self.event, self.event_group_ids,
                             self.n_groups, indices, weights)
