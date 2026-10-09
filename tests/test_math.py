import numpy as np
import pytest

from kesurv._curve_distance import agreement_counts, agreement_reference
from kesurv._forest_weights import ForestWeights
from kesurv._weighted_km import WeightedKM


class FakeForest:
    def __init__(self):
        self.estimators_samples_ = [np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])]

    def apply(self, X):
        return np.asarray(X, dtype=int)


def test_forest_kernel_uses_bootstrap_multiplicity():
    # Each row is a sample's leaf ID in two trees. Subject 0 is repeated in
    # tree 1 and subject 1 in tree 2.
    X = np.array([[1, 1], [1, 1], [2, 2]])
    kernel = ForestWeights(FakeForest(), X)
    indices, weights = kernel.one(np.array([1, 1]))
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(weights, [0.5, 0.5, 0.0])
    assert weights.sum() == pytest.approx(1)


def test_weighted_km_ties_censoring_and_zero_time():
    event = np.array([True, False, True, True])
    time = np.array([0.0, 1.0, 1.0, 2.0])
    km = WeightedKM(event, time)
    indices = np.arange(4)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    # At t=0, hazard=0.1/1. At t=1, hazard=0.3/0.9. At t=2,
    # the last at-risk subject dies.
    expected = np.array([0.9, 0.6, 0.0])
    np.testing.assert_allclose(km.predict(indices, weights, backend="numpy"), expected)


def test_weighted_km_numba_matches_reference():
    pytest.importorskip("numba")
    event = np.array([True, False, True, True])
    time = np.array([0.0, 1.0, 1.0, 2.0])
    km = WeightedKM(event, time)
    indices = np.arange(4)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        km.predict(indices, weights, backend="numba"),
        km.predict(indices, weights, backend="numpy"),
    )


def test_weighted_km_all_selected_censored_is_flat():
    km = WeightedKM(np.array([False, True]), np.array([1.0, 2.0]))
    result = km.predict(np.array([0]), np.array([1.0]), backend="numpy")
    np.testing.assert_allclose(result, [1.0])


def test_agreement_numba_matches_reference_with_uneven_grid():
    pytest.importorskip("numba")
    train = np.array([
        [[1, 0.8, 0.7], [1, 0.5, 0.4], [1, 0.9, 0.8]],
        [[1, 0.8, 0.6], [1, 0.7, 0.3], [1, 0.4, 0.2]],
    ], dtype=np.float32)
    query = np.array([[1, 0.8, 0.7], [1, 0.8, 0.6]], dtype=np.float32)
    widths = np.array([1.0, 3.0])
    indices = np.array([0, 1, 2])
    expected = agreement_reference(train, query, indices, widths, cutoff=0.7)
    observed = agreement_counts(train, query, indices, widths, 0.7, 2,
                                backend="numba", full_counts=True)
    np.testing.assert_array_equal(observed, expected)

