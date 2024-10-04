"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
"""

from typing import Optional, Tuple
from numpy import ndarray
from scalar_pdf import Value, Values, Values2D

import numpy as np
from scalar_pdf import ScalarPDF, check_data
from stats_tools import logistic, logit, guard_prob, weighted_mean


DEFAULT_THETA = 0.5


class BernoulliPDF(ScalarPDF):

    def __init__(self, theta: Value = DEFAULT_THETA):
        """
        Initialises the Bernoulli distribution(s).

        Input:
            - theta (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta)

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_THETA,)

    def mean(self) -> Value:
        theta = self.parameters()[0]
        return theta

    def variance(self) -> Value:
        theta = self.parameters()[0]
        return theta * (1 - theta)

    def log_prob(self, X: Value) -> Value:
        theta = guard_prob(self.parameters()[0])
        return X * np.log(theta) + (1 - X) * np.log(1 - theta)

    def _internal_parameters(self, theta: Value) -> Values:
        theta = guard_prob(theta)
        eta = logit(theta)
        return (eta,)

    def _distributional_parameters(self, eta: Value) -> Values:
        theta = logistic(eta)
        return (theta,)

    def _estimate_parameters(
        self, X: Value, W: Optional[Value] = None, **kwargs: dict
    ) -> Values:
        X, W = check_data(X, W)
        theta = weighted_mean(W, X)
        return (theta,)

    def _internal_gradient(self, X: Value) -> Values:
        # d L / d eta = X - E[X]
        mu = self.mean()
        return (X - mu,)

    def _internal_negHessian(self, X: Value) -> Values2D:
        # - d^2 L / d eta^2 = Var[X]
        v = self.variance()
        return ((v,),)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BernoulliPDF()
    assert bd.parameters() == (DEFAULT_THETA,)
    assert bd.mean() == DEFAULT_THETA
    assert bd.variance() == DEFAULT_THETA * (1 - DEFAULT_THETA)

    t_theta = bd.parameters()
    t_eta = bd._internal_parameters(*t_theta)
    t_theta2 = bd._distributional_parameters(*t_eta)
    t_eta2 = bd._internal_parameters(*t_theta2)
    assert t_theta2 == t_theta
    assert t_eta2 == t_eta
    print("Passed default parameter tests!")

    # Test specified parameter
    theta = 0.123456
    bd = BernoulliPDF(theta)
    assert bd.parameters() == (theta,)
    assert bd.mean() == theta
    assert bd.variance() == theta * (1 - theta)
    print("Passed specified parameter tests!")

    # Test fitting 1 observation - be careful with 0 or 1!
    for X in [1e-3, 0.9, 0.5, 0.1, 1, 0]:
        bd = BernoulliPDF()
        res = bd.fit(X)
        assert np.abs(bd.parameters()[0] - X) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    for n in range(2, 11):
        X = np.random.randint(0, 2, n)
        bd = BernoulliPDF()
        res = bd.fit(X)
        assert np.abs(bd.parameters()[0] - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")
