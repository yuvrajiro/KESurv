"""Compare NumPy, Numba, and optional Cython similarity vote kernels."""

from time import perf_counter

import numpy as np

from kesurv._curve_distance import agreement_counts

try:
    from _cython_trial import agreement_cython
except ImportError:
    agreement_cython = None


def main():
    rng = np.random.default_rng(9)
    n_models, n_train, n_timepoints = 4, 2000, 128
    train = np.ascontiguousarray(
        np.cumprod(rng.uniform(0.98, 1, size=(n_models, n_train, n_timepoints)),
                   axis=2).astype(np.float32)
    )
    query = np.ascontiguousarray(train[:, 0, :])
    indices = np.arange(n_train, dtype=np.int64)
    widths = np.full(n_timepoints - 1, 1 / (n_timepoints - 1), dtype=np.float64)
    cutoff = 0.05
    minimum = 2
    runs = {}
    methods = {
        "numpy": lambda: agreement_counts(train, query, indices, widths,
                                            cutoff, minimum, backend="numpy") >= minimum,
    }
    try:
        import numba  # noqa: F401
        methods["numba"] = lambda: agreement_counts(
            train, query, indices, widths, cutoff, minimum, backend="numba"
        ) >= minimum
    except ImportError:
        pass
    if agreement_cython is not None:
        methods["cython"] = lambda: np.asarray(
            agreement_cython(train, query, indices, widths, cutoff, minimum),
            dtype=bool,
        )
    for name, method in methods.items():
        method()  # warm Numba and Cython
        started = perf_counter()
        for _ in range(50):
            selected = method()
        runs[name] = (perf_counter() - started) / 50
        np.testing.assert_array_equal(selected, methods["numpy"]())
    for name, seconds in runs.items():
        print(f"{name:>6}: {seconds * 1000:.3f} ms/query")


if __name__ == "__main__":
    main()
