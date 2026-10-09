import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import StepFunction
from sksurv.util import Surv

from kesurv import KESurv


class SimpleCurves(BaseEstimator):
    def fit(self, X, y):
        self.scale_ = float(np.max(y[y.dtype.names[1]]))
        return self

    def predict_survival_function(self, X):
        curves = []
        for row in X:
            value = 0.8 if row[0] < 0.5 else 0.4
            curves.append(StepFunction([0.0, self.scale_], [1.0, value],
                                       domain=(0, np.inf)))
        return curves


class ContinuousCurves(BaseEstimator):
    def fit(self, X, y):
        self.scale_ = float(np.max(y[y.dtype.names[1]]))
        return self

    def predict_survival_function(self, X):
        return [
            StepFunction(
                np.array([0.0, self.scale_ / 2, self.scale_]),
                np.array([1.0, np.exp(-abs(float(row[0]))),
                          np.exp(-abs(float(row[0])))]),
                domain=(0, np.inf),
            )
            for row in X
        ]


class BareCustomEstimator:
    """Only the two required methods; no sklearn inheritance or get_params."""

    def fit(self, X, y):
        self.times_ = np.array([0.0, float(np.max(y[y.dtype.names[1]]))])
        return self

    def predict_survival_function(self, X):
        return [StepFunction(self.times_, np.array([1.0, 0.7]),
                             domain=(0, np.inf)) for _ in X]


@pytest.fixture
def sample():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(36, 2))
    event = rng.random(36) > 0.25
    time = rng.uniform(0.2, 5, size=36)
    return X, Surv.from_arrays(event, time)


def make_model(**kwargs):
    options = dict(
        base_estimators=[("simple", SimpleCurves()),
                         ("tree", SurvivalTree(max_depth=2, random_state=4))],
        forest=RandomSurvivalForest(n_estimators=8, max_depth=3, random_state=3),
        min_agreement=1,
        n_timepoints=16,
        backend="numpy",
    )
    options.update(kwargs)
    return KESurv(**options)


def test_fit_predict_and_diagnostics(sample):
    X, y = sample
    model = make_model().fit(X, y)
    prediction = model.predict_survival_function(X[:4], return_array=True)
    assert prediction.shape == (4, len(model.unique_times_))
    assert np.all((0 <= prediction) & (prediction <= 1))
    assert np.all(np.diff(prediction, axis=1) <= 1e-12)
    curves = model.predict_survival_function(X[:4])
    np.testing.assert_allclose([curve(0.0) for curve in curves], [1.0] * 4)
    np.testing.assert_allclose([curve(1000.0) for curve in curves], prediction[:, -1])
    weights = model.predict_weights(X[:4])
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    info = model.predict_neighbor_info(X[:2])
    assert len(info) == 2
    assert set(info[0]) == {"candidate_indices", "forest_weights",
                            "agreement_counts", "selected_indices",
                            "selected_weights", "fallback"}
    assert model.predict_rmst(X[:2], 2.0).shape == (2,)
    assert model.kneighbors(X[:2], 3)[0].shape == (2, 3)


def test_infinite_epsilon_matches_forest_kernel(sample):
    X, y = sample
    model = make_model(epsilon=np.inf).fit(X, y)
    query = X[:3]
    weights = model.predict_weights(query)
    leaves = model._forest_weights.apply(query)
    for row in range(3):
        _, expected = model._forest_weights.one(leaves[row])
        np.testing.assert_allclose(weights[row], expected)


def test_zero_selected_mass_falls_back(sample):
    X, y = sample
    model = make_model(base_estimators=[("continuous", ContinuousCurves())],
                       epsilon=0.0, min_agreement=1).fit(X, y)
    query = np.array([[10.0, 0.0]])
    info = model.predict_neighbor_info(query)[0]
    assert info["fallback"]
    np.testing.assert_array_equal(info["selected_indices"], info["candidate_indices"])
    np.testing.assert_allclose(info["selected_weights"], info["forest_weights"])


def test_custom_grid_and_validation(sample):
    X, y = sample
    model = KESurv(base_estimators=[("simple", SimpleCurves())],
                   similarity_times=[0.0, 0.5, 2.0, 4.0],
                   forest=RandomSurvivalForest(n_estimators=4, random_state=2),
                   backend="numpy").fit(X, y)
    np.testing.assert_array_equal(model.similarity_times_, [0, 0.5, 2, 4])
    assert model.predict_survival(X[:2], [0, 1, 9]).shape == (2, 3)
    with pytest.raises(ValueError, match="min_agreement"):
        KESurv(base_estimators=[("simple", SimpleCurves())],
               min_agreement=2).fit(X, y)


def test_numba_and_numpy_predictions_agree(sample):
    pytest.importorskip("numba")
    X, y = sample
    model = make_model().fit(X, y)
    reference = model.predict_survival_function(X[:5], return_array=True)
    model.backend = "numba"
    fast = model.predict_survival_function(X[:5], return_array=True)
    np.testing.assert_allclose(fast, reference, atol=1e-12)


def test_minimal_custom_estimator_and_copy(sample):
    X, y = sample
    custom = BareCustomEstimator()
    model = make_model(base_estimators=[("custom", custom)]).fit(X, y)
    assert not hasattr(custom, "times_")
    assert model.predict_survival_function(X[:2], return_array=True).shape[0] == 2


def test_invalid_custom_curve_rejected(sample):
    X, y = sample

    class Increasing(BareCustomEstimator):
        def predict_survival_function(self, X):
            return [StepFunction(np.array([0.0, 1.0]), np.array([0.2, 0.8]),
                                 domain=(0, np.inf)) for _ in X]

    with pytest.raises(ValueError, match="increasing survival curve"):
        make_model(base_estimators=[("bad", Increasing())]).fit(X, y)
