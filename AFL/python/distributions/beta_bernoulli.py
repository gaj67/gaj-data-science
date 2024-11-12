"""
This module implements the beta-Bernoulli distribution.

The distributional parameters are the beta shape parameters,
namely alpha and beta.

The link parameter is chosen as eta = logit(mu) = log(alpha/beta),
which is not a natural parameter. The independent parameter is
chosen to be psi = alpha. 
"""

import numpy as np

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Vector,
)

from .core.distribution import StandardDistribution, set_link
from .core.link_models import LogRatioLink2a


#################################################################
# Beta-Bernoulli distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_link(LogRatioLink2a)
class BetaBernoulliDistribution(StandardDistribution):
    """
    Implements the beta-Bernoulli probability distribution for a
    binary response variate, X.

    The shape parameters, alpha and beta, govern the probability
    that X=1.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the beta-Bernoulli distribution(s).

        Input:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (DEFAULT_ALPHA, DEFAULT_BETA)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        alpha, beta = params
        return np.all(alpha > 0) and np.all(beta > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        alpha, beta = self.get_parameters()
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        theta = self.mean()
        return theta * (1 - theta)

    # -----------------------------
    # GradientOptimisable interface

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self.get_parameters()
        return (
            variate * np.log(alpha)
            + (1 - variate) * np.log(beta)
            - np.log(alpha + beta)
        )

    def compute_gradients(self, variate: Vector) -> Values:
        alpha, beta = self.get_parameters()
        g_alpha = variate / alpha - 1 / (alpha + beta)
        g_beta = (1 - variate) / beta - 1 / (alpha + beta)
        return (g_alpha, g_beta)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BetaBernoulliDistribution()
    assert len(bd.get_parameters()) == 2
    assert bd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = bd.mean()
    assert mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
    assert bd.variance() == mu * (1 - mu)
    print("Passed default parameter tests!")

    # Test fitting multiple observations
    X = np.array([1, 1, 0, 1])
    bd = BetaBernoulliDistribution()
    res = bd.fit(X)
    assert np.abs(bd.mean() - np.mean(X)) < 1e-6
    _alpha, _beta = bd.get_parameters()
    assert np.abs(_alpha / _beta - 3) < 1e-5
    print("Passed fitting multiple observations test!")

    # Test regression without intercept/bias
    br = BetaBernoulliDistribution().regressor()
    X = [1, 0]
    Z = [1, -1]
    res = br.fit(X, Z, max_iters=1000)
    phi, psi = br.get_parameters()
    assert len(phi) == 1
    mu = br.mean(Z)
    for x, m in zip(X, mu):
        assert np.abs(m - x) < 1e-2
    print("Passed fitting simple regression test!")

    # Test regression with intercept/bias
    br = BetaBernoulliDistribution().regressor()
    X = [1, 0, 1, 1, 0]
    Z = [(1, z) for z in [1, 1, -1, -1, -1]]
    res = br.fit(X, Z)
    mu = br.mean((1, 1))
    assert np.abs(mu - 0.5) < 1e-5
    mu = br.mean([(1, -1)])
    assert np.abs(mu - 2 / 3) < 1e-5
    print("Passed fitting simple regression test!")
