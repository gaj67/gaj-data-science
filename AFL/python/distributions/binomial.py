"""
This module implements the binomial distribution for
a pre-specified number of Bernoulli trials.

The distributional parameters are n, the number of trials,
and theta, the probability of the event of interest.

The link parameter is the natural parameter, eta = logit(theta).
"""

import numpy as np
from scipy.special import loggamma


if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
    mean_value,
)

from .core.parameterised import UNSPECIFIED_VECTOR, guard_prob
from .core.distribution import StandardDistribution, set_link_model, add_link_model
from .core.link_models import SwapLink20, LogitLink21
from .core.optimiser import Data, Controls


#################################################################
# Binomial distribution

DEFAULT_THETA = 0.5


@add_link_model(LogitLink21)  # eta = logit(theta) <-> theta = logistic(eta)
@set_link_model(SwapLink20)  # swap order of (n, theta) <-> (theta, n)
class BinomialDistribution(StandardDistribution):
    """
    Implements the binomial probability distribution for a non-negative
    response variate, X.

    The first parameter, n, governs the number of Bernoulli trials.
    The second parameter, theta, governs the probability that any
    independent trial is a special event.
    """

    def __init__(self, n: int, theta: Value = DEFAULT_THETA):
        """
        Initialises the binomial distribution(s).

        Input:
            - n (int): The number of trials.
            - theta (float or ndarray): The probability value(s).
        """
        super().__init__(n, theta)

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (0, DEFAULT_THETA)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        n, theta = params
        # TODO check n >= 0
        return np.all(theta > 0) and np.all(theta < 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        n, theta = self.get_parameters()
        return n * theta

    def variance(self) -> Value:
        n, theta = self.get_parameters()
        return n * theta * (1 - theta)

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        n = self.get_parameters()[0]
        theta = guard_prob(mean_value(data.weights, data.variate) / n)
        return (n, theta)

    def compute_scores(self, variate: Vector) -> Vector:
        n, theta = self.get_parameters()
        log_c_n_x = loggamma(n + 1) - loggamma(variate + 1) - loggamma(n - variate + 1)
        return log_c_n_x + variate * np.log(theta) + (n - variate) * np.log(1 - theta) 

    def compute_gradients(self, variate: Vector) -> Values:
        n, theta = self.get_parameters()
        grads = (variate - n * theta) / (theta * (1 - theta))
        return (0, grads)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        # Force to be invertible, since n is constant
        n, theta = self.get_parameters()
        v = n / (theta * (1 - theta))
        return ((1, 0), (0, v))


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    N = 5
    bd = BinomialDistribution(N)
    assert bd.get_parameters() == (N, DEFAULT_THETA)
    assert bd.mean() == N * DEFAULT_THETA
    assert bd.variance() == N * DEFAULT_THETA * (1 - DEFAULT_THETA)
    print("Passed default parameter tests!")

    # Test specified parameter
    THETA = 0.123456
    bd = BinomialDistribution(1, THETA)
    assert bd.get_parameters() == (1, THETA)
    assert bd.mean() == THETA
    assert bd.variance() == THETA * (1 - THETA)
    print("Passed specified parameter tests!")

    # Test fitting 1 observation - be careful with 0 or n!
    for X in range(N + 1):
        bd = BinomialDistribution(N)
        res = bd.fit(X)
        assert np.abs(bd.mean() - X) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    X = [1, 1, 0, 5, 3, 2, 4, 4, 1, 0]
    bd = BinomialDistribution(N, THETA)
    res = bd.fit(X)
    assert np.abs(bd.mean() - np.mean(X)) < 1e-6

    for n in range(2, 11):
        X = np.random.randint(0, N + 1, n)
        bd = BinomialDistribution(N)
        res = bd.fit(X)
        assert np.abs(bd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")

    # ---------------------------------------------------
    # Test default parameter
    br = BinomialDistribution(N).regressor()
    assert br.get_parameters() == (UNSPECIFIED_VECTOR, N)
    print("Passed default regression parameter tests!")

    # Test fitting two groups of multiple observations
    Xm1 = [1, 2, 3, 1, 0, 0, 1, 2, 1, 0]
    Zm1 = [-1] * len(Xm1)
    Xp1 = [N - x for x in Xm1]
    Zp1 = [1] * len(Xp1)
    X = Xm1 + Xp1
    Z = Zm1 + Zp1
    res = br.fit(X, Z, score_tol=1e-12)
    assert res["converged"]
    mu_p1 = br.mean(1)
    assert np.abs(mu_p1 - np.mean(Xp1)) < 1e-6
    mu_m1 = br.mean(-1)
    assert np.abs(mu_m1 - np.mean(Xm1)) < 1e-6
    print("Passed fitting grouped observations tests!")
