# Performance and tuning

The main accuracy–speed controls are the similarity grid, forest size, and prediction backend. Start with a modest grid, then compare predictions at higher resolutions on a validation set.

| Parameter | Effect |
| --- | --- |
| `similarity_time` | Upper time limit for comparing predicted curves |
| `n_timepoints` | Number of points in the uniform similarity grid, including zero and horizon |
| `similarity_times` | Explicit grid when important times are irregular |
| `batch_size` | Number of query subjects passed to base learners together |
| `backend` | `"auto"`, `"numpy"`, or `"numba"` for agreement and weighted KM kernels |

`backend="auto"` uses Numba when installed, otherwise NumPy. The first Numba prediction includes compilation, so compare **warmed** runs. If portability or predictable cold-start time is more important, choose `backend="numpy"`.

```python
model = KESurv(
    base_estimators=base_estimators,
    similarity_time=5.0,
    n_timepoints=64,
    batch_size=128,
    backend="auto",
)
```

The package caches training leaf IDs, bootstrap multiplicities, base-model survival predictions on the similarity grid, and observed-time groups. Query comparisons only visit forest-supported candidates. Forest fitting and base-model prediction can still dominate runtime.

## Benchmark locally

```console
python benchmarks/benchmark_prediction.py
```

This synthetic benchmark warms each backend, checks NumPy and Numba output agreement, and reports time per batch. Its numbers depend on hardware, dataset, forest settings, and installed packages. A separate Cython prototype lives in `benchmarks/`; it is an experiment, not an installation requirement. The weighted counting design was informed by [fastkme](https://github.com/yuvrajiro/fastkme).

## Practical sequence

1. Set a medically meaningful `similarity_time` and test whether the horizon changes neighbor selection.
2. Compare `n_timepoints=64` and `128` on held-out examples, including `predict_neighbor_info`.
3. Increase forest trees or tune base models only after checking the grid.
4. Benchmark warmed prediction with the expected query batch size.
