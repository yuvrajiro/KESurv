"""Compare prediction kernels on synthetic right-censored data.

Run from an editable install or with PYTHONPATH=src. Results vary by machine.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from sksurv.util import Surv

from kesurv import KESurv


def main(n_train=1000, n_query=100, n_timepoints=128):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(n_train + n_query, 8))
    event_times = rng.exponential(scale=np.exp(-0.3 * X[:, 0] + 0.2 * X[:, 1]))
    censor_times = rng.exponential(scale=2.0, size=len(X))
    event = event_times <= censor_times
    y = Surv.from_arrays(event[:n_train], np.minimum(event_times, censor_times)[:n_train])
    model = KESurv(
        base_estimators=[
            ("tree1", SurvivalTree(max_depth=4, random_state=1)),
            ("tree2", SurvivalTree(max_depth=6, random_state=2)),
        ],
        forest=RandomSurvivalForest(n_estimators=32, max_depth=6, random_state=3),
        epsilon=0.05,
        min_agreement=1,
        n_timepoints=n_timepoints,
        batch_size=32,
        backend="numpy",
    ).fit(X[:n_train], y)
    queries = X[n_train:]
    timings = {}
    outputs = {}
    try:
        import numba  # noqa: F401
        backends = ("numpy", "numba")
    except ImportError:
        backends = ("numpy",)
    for backend in backends:
        model.backend = backend
        model.predict_survival_function(queries[:1], return_array=True)  # warm JIT
        started = perf_counter()
        outputs[backend] = model.predict_survival_function(queries, return_array=True)
        timings[backend] = perf_counter() - started
    if "numba" in outputs:
        np.testing.assert_allclose(outputs["numpy"], outputs["numba"], atol=1e-10)
    print(f"n_train={n_train}, n_query={n_query}, n_timepoints={n_timepoints}")
    for backend, seconds in timings.items():
        print(f"{backend:>5}: {seconds:.3f} s; {n_query / seconds:.1f} queries/s")
    if "numba" in timings:
        print(f"speedup: {timings['numpy'] / timings['numba']:.2f}x")


if __name__ == "__main__":
    main()
