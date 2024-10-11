"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
"""

from typing import Optional

import numpy as np

from core.value_types import (
    Value,
    Values,
    is_divergent,
    is_scalar,
    to_value,
)

from core.distribution import Distribution

from core.fitter import Fittable, Controls, Results


from stats_tools import guard_prob, weighted_mean


DEFAULT_THETA = 0.5


#################################################################
# Bernoulli distribution


class BernoulliDistribution(Distribution, Fittable):
    """
    Implements the Bernoulli probability distribution for a binary
    response variate, X.

    The sole parameter, theta, governs the probability that X=1.

    The natural parameter is eta = logit(theta) with natural variate X.
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

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        for value in params:
            if is_divergent(value) or np.any(value < 0) or np.any(value > 1):
                return False
        return True

    def mean(self) -> Value:
        theta = self.parameters()[0]
        return theta

    def variance(self) -> Value:
        theta = self.parameters()[0]
        return theta * (1 - theta)

    def log_prob(self, data: Value) -> Value:
        theta = guard_prob(self.parameters()[0])
        return data * np.log(theta) + (1 - data) * np.log(1 - theta)

    def fit(
        self,
        data: Value,
        weights: Optional[Value] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - data (float-like or array-like): The value(s) of the observation(s).
            - weights (float-like or array-like, optional): The weight(s) of the
                observation(s).
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output. See optimise_parameters().
        """
        data = to_value(data)
        if weights is None:
            weights = 1.0 if is_scalar(data) else np.ones(len(data))
        theta = weighted_mean(weights, data)
        self.set_parameters(theta)
        score = np.sum(weights * self.log_prob(data)) / np.sum(weights)
        return {
            "score": score,
            "score_tol": 0.0,
            "num_iters": 0,
        }


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BernoulliDistribution()
    assert bd.parameters() == (DEFAULT_THETA,)
    assert bd.mean() == DEFAULT_THETA
    assert bd.variance() == DEFAULT_THETA * (1 - DEFAULT_THETA)
    print("Passed default parameter tests!")

    # Test specified parameter
    THETA = 0.123456
    bd = BernoulliDistribution(THETA)
    assert bd.parameters() == (THETA,)
    assert bd.mean() == THETA
    assert bd.variance() == THETA * (1 - THETA)
    print("Passed specified parameter tests!")

    X = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    bd = BernoulliDistribution(THETA)
    res = bd.fit(X)
    assert np.abs(bd.mean() - np.mean(X)) < 1e-6

    # Test fitting 1 observation - be careful with 0 or 1!
    for X in [1e-3, 0.9, 0.5, 0.1, 1, 0]:
        bd = BernoulliDistribution()
        res = bd.fit(X)
        assert np.abs(bd.mean() - X) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    for n in range(2, 11):
        X = np.random.randint(0, 2, n)
        bd = BernoulliDistribution()
        res = bd.fit(X)
        assert np.abs(bd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")
