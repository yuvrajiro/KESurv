"""Minimal custom base learner compatible with KESurv.

Replace this population Kaplan-Meier learner with any fitted model that
returns one survival callable per input row. Subclassing BaseEstimator makes
the learner cloneable for cross-validation and hyperparameter tuning.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator


class PopulationKM(BaseEstimator):
    def fit(self, X, y):
        event = y[y.dtype.names[0]]
        time = y[y.dtype.names[1]]
        self.unique_times_, survival = kaplan_meier_estimator(event, time)
        if self.unique_times_[0] > 0:
            knots = np.concatenate(([0.0], self.unique_times_))
            values = np.concatenate(([1.0], survival))
        else:
            knots, values = self.unique_times_, survival
        self.curve_ = StepFunction(knots, values, domain=(0, np.inf))
        return self

    def predict_survival_function(self, X):
        return [self.curve_] * len(X)

