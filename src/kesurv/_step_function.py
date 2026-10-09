"""Callable, right-continuous survival curve with a stable tail."""

from __future__ import annotations

import numpy as np


class SurvivalStepFunction:
    """A step function equal to 1 before the first event and flat afterward."""

    def __init__(self, times, values):
        self.x = np.asarray(times, dtype=np.float64)
        self.y = np.asarray(values, dtype=np.float64)
        if self.x.ndim != 1 or self.x.shape != self.y.shape or len(self.x) == 0:
            raise ValueError("times and values must be matching nonempty vectors")

    def __call__(self, times):
        requested = np.asarray(times, dtype=np.float64)
        if not np.isfinite(requested).all() or (requested < 0).any():
            raise ValueError("survival times must be finite and nonnegative")
        positions = np.searchsorted(self.x, requested, side="right") - 1
        result = np.ones(requested.shape, dtype=np.float64)
        active = positions >= 0
        result[active] = self.y[positions[active]]
        return float(result) if result.ndim == 0 else result

