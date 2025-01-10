"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
The link parameter is the natural parameter, eta = logit(theta).
"""

import numpy as np

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
from .core.distribution import StandardDistribution, set_link
from .core.link_models import LogitLink11
from .core.optimiser import Data, Controls


#################################################################
# Bernoulli distribution

DEFAULT_THETA = 0.5


@set_link(LogitLink11)
class BernoulliDistribution(StandardDistribution):
    """
    Implements the Bernoulli probability distribution for a binary
    response variate, X.

    The sole parameter, theta, governs the probability that X=1.
    """

    def __init__(self, theta: Value = DEFAULT_THETA):
        """
        Initialises the Bernoulli distribution(s).

        Input:
            - theta (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta)

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (DEFAULT_THETA,)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 1 or not super().check_parameters(*params):
            return False
        theta = params[0]
        return np.all(theta > 0) and np.all(theta < 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        theta = self.get_parameters()[0]
        return theta

    def variance(self) -> Value:
        theta = self.get_parameters()[0]
        return theta * (1 - theta)

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        theta = guard_prob(mean_value(data.weights, data.variate))
        print("DEBUG[BernoulliDistribution.compute_estimates]: theta=", theta)
        return (theta,)

    def compute_scores(self, variate: Vector) -> Vector:
        theta = self.get_parameters()[0]
        return variate * np.log(theta) + (1 - variate) * np.log(1 - theta)

    def compute_gradients(self, variate: Vector) -> Values:
        theta = self.get_parameters()[0]
        grads = (variate - theta) / (theta * (1 - theta))
        return (grads,)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        # Use expected value
        theta = self.get_parameters()[0]
        v = 1.0 / (theta * (1 - theta))
        return ((v,),)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BernoulliDistribution()
    assert bd.get_parameters() == (DEFAULT_THETA,)
    assert bd.mean() == DEFAULT_THETA
    assert bd.variance() == DEFAULT_THETA * (1 - DEFAULT_THETA)
    print("Passed default parameter tests!")

    # Test specified parameter
    THETA = 0.123456
    bd = BernoulliDistribution(THETA)
    assert bd.get_parameters() == (THETA,)
    assert bd.mean() == THETA
    assert bd.variance() == THETA * (1 - THETA)
    print("Passed specified parameter tests!")

    # Test fitting 1 observation - be careful with 0 or 1!
    for X in [1e-3, 0.9, 0.5, 0.1, 1, 0]:
        bd = BernoulliDistribution()
        res = bd.fit(X)
        assert np.abs(bd.mean() - X) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    X = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    bd = BernoulliDistribution(THETA)
    res = bd.fit(X)
    assert np.abs(bd.mean() - np.mean(X)) < 1e-6

    for n in range(2, 11):
        X = np.random.randint(0, 2, n)
        bd = BernoulliDistribution()
        res = bd.fit(X)
        assert np.abs(bd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")

    # ---------------------------------------------------
    # Test default parameter
    br = BernoulliDistribution().regressor()
    assert br.get_parameters() == (UNSPECIFIED_VECTOR,)
    print("Passed default regression parameter tests!")

    # Test fitting two groups of multiple observations
    Xp1 = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    Zp1 = [1] * len(Xp1)
    Xm1 = [1 - x for x in Xp1]
    Zm1 = [-1] * len(Xm1)
    X = Xp1 + Xm1
    Z = Zp1 + Zm1
    res = br.fit(X, Z, score_tol=1e-12)
    assert res["converged"]
    mu_p1 = br.mean(1)
    assert np.abs(mu_p1 - np.mean(Xp1)) < 1e-6
    mu_m1 = br.mean(-1)
    assert np.abs(mu_m1 - np.mean(Xm1)) < 1e-6
    print("Passed fitting grouped observations tests!")
