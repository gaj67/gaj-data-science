"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
"""

import numpy as np

from scalar_pdf import (
    ScalarPDF,
    Value,
    Values,
    Values2D,
    check_transformations,
)
from stats_tools import logistic, logit, guard_prob, weighted_mean


DEFAULT_THETA = 0.5


class BernoulliPDF(ScalarPDF):
    """
    Implements the Bernoulli probability distribution for a binary
    response variate, X.
    """

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

    def log_prob(self, data: Value) -> Value:
        theta = guard_prob(self.parameters()[0])
        return data * np.log(theta) + (1 - data) * np.log(1 - theta)

    def _internal_parameters(self, *theta: Values) -> Values:
        theta = guard_prob(theta[0])
        eta = logit(theta)
        return (eta,)

    def _distributional_parameters(self, *psi: Values) -> Values:
        eta = psi[0]
        theta = logistic(eta)
        return (theta,)

    def _estimate_parameters(
        self, data: Value, weights: Value, **kwargs: dict
    ) -> Values:
        theta = weighted_mean(weights, data)
        return (theta,)

    def _internal_gradient(self, data: Value) -> Values:
        # d L / d eta = X - E[X]
        mu = self.mean()
        return (data - mu,)

    def _internal_neg_hessian(self, data: Value) -> Values2D:
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
    print("Passed default parameter tests!")

    # Test specified parameter
    THETA = 0.123456
    bd = BernoulliPDF(THETA)
    assert bd.parameters() == (THETA,)
    assert bd.mean() == THETA
    assert bd.variance() == THETA * (1 - THETA)
    print("Passed specified parameter tests!")

    assert check_transformations(bd)
    print("Passed parameter transformations tests!")

    # Test fitting 1 observation - be careful with 0 or 1!
    for value in [1e-3, 0.9, 0.5, 0.1, 1, 0]:
        bd = BernoulliPDF()
        res = bd.fit(value)
        assert np.abs(bd.mean() - value) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    for n in range(2, 11):
        values = np.random.randint(0, 2, n)
        bd = BernoulliPDF()
        res = bd.fit(values)
        assert np.abs(bd.mean() - np.mean(values)) < 1e-6
    print("Passed fitting multiple observations tests!")
