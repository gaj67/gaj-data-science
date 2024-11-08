"""
This module implements the Poisson distribution.

The distributional parameter, lambda, is also the mean, mu.
The link parameter is the natural parameter, eta = log(lambda).
"""

import numpy as np
from scipy.special import gamma

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
)

from .core.parameterised import UNSPECIFIED_VECTOR
from .core.distribution import StandardDistribution, set_link
from .core.link_models import LogLink1


#################################################################
# Poisson distribution


DEFAULT_LAMBDA = 1.0


@set_link(LogLink1)
class PoissonDistribution(StandardDistribution):
    """
    Implements the Poisson probability distribution for a binary
    response variate, X.

    The sole parameter, lambda, governs the probability of
    X = 0, 1, 2, ... .
    """

    def __init__(self, _lambda: Value = DEFAULT_LAMBDA):
        """
        Initialises the Poisson distribution(s).

        Input:
            - _lambda (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(_lambda)

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (DEFAULT_LAMBDA,)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        if not super().check_parameters(*params):
            return False
        _lambda = params[0]
        return np.all(_lambda > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        _lambda = self.get_parameters()[0]
        return _lambda

    def variance(self) -> Value:
        _lambda = self.get_parameters()[0]
        return _lambda

    # -----------------------------
    # GradientOptimisable interface

    #    def compute_estimate(self, data: Data, controls: Controls) -> Values:
    #        _lambda = guard_pos(mean_value(data.weights, data.variate))
    #        return (_lambda,)

    def compute_estimates(self, variate: Vector) -> Values:
        ind = variate > 0
        _lambda = variate[ind]
        return ind, (_lambda,)

    def compute_scores(self, variate: Vector) -> Vector:
        _lambda = self.get_parameters()[0]
        return variate * np.log(_lambda) - _lambda - np.log(gamma(variate + 1))

    def compute_gradients(self, variate: Vector) -> Values:
        _lambda = self.get_parameters()[0]
        g_lam = (variate - _lambda) / _lambda
        return (g_lam,)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        # Use expected value
        _lambda = self.get_parameters()[0]
        v_lam = 1 / _lambda
        return ((v_lam,),)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    pd = PoissonDistribution()
    assert pd.get_parameters() == (DEFAULT_LAMBDA,)
    assert pd.mean() == DEFAULT_LAMBDA
    assert pd.variance() == DEFAULT_LAMBDA
    print("Passed default parameter tests!")

    # Test specified parameter
    LAMBDA = 0.123456
    pd = PoissonDistribution(LAMBDA)
    assert pd.get_parameters() == (LAMBDA,)
    assert pd.mean() == LAMBDA
    assert pd.variance() == LAMBDA
    print("Passed specified parameter tests!")

    # Test fitting 1 observation
    for X in [0, 1, 10, 100]:
        pd = PoissonDistribution()
        res = pd.fit(X)
        assert np.abs(pd.mean() - X) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    for n in range(2, 11):
        X = np.random.randint(0, 100, n)
        pd = PoissonDistribution()
        res = pd.fit(X)
        assert np.abs(pd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")

    # ---------------------------------------------------

    # Test default parameter
    pr = PoissonDistribution().regressor()
    assert pr.get_parameters() == (UNSPECIFIED_VECTOR,)
    print("Passed default regression parameter tests!")

    # Test fitting two groups of multiple observations
    Xp1 = [1, 2, 3, 1, 0, 0, 5, 4, 1, 0]
    Zp1 = [1] * len(Xp1)
    Xm1 = np.array(Xp1) / np.mean(Xp1) ** 2
    Zm1 = [-1] * len(Xm1)
    X = np.concat((Xp1, Xm1))
    Z = np.concat((Zp1, Zm1))
    res = pr.fit(X, Z, score_tol=1e-12)
    assert res["converged"]
    mu_p1 = pr.mean(1)
    assert np.abs(mu_p1 - np.mean(Xp1)) < 1e-6
    mu_m1 = pr.mean(-1)
    assert np.abs(mu_m1 - np.mean(Xm1)) < 1e-6
    print("Passed fitting grouped observations tests!")

    pr = PoissonDistribution().regressor()
    Zp1 = [(1, 1)] * len(Xp1)
    Zm1 = [(1, -1)] * len(Xm1)
    Z = np.concat((Zp1, Zm1))
    res = pr.fit(X, Z, score_tol=1e-12)
    assert res["converged"]
    phi0 = pr.get_parameters()[0][0]
    assert np.abs(phi0) < 1e-6
    mu_p1 = pr.mean([1, 1])
    assert np.abs(mu_p1 - np.mean(Xp1)) < 1e-5
    mu_m1 = pr.mean([1, -1])
    assert np.abs(mu_m1 - np.mean(Xm1)) < 1e-5
    print("Passed fitting multiple regression tests!")
