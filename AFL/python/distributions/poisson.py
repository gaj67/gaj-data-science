"""
This module implements the Poisson distribution.

The distributional parameter, lambda, is also the mean, mu.
The link parameter is the natural parameter, eta = log(lambda).
"""

from typing import Optional

import numpy as np
from numpy.linalg import solve
from scipy.special import gamma

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
    VectorLike,
    is_divergent,
    is_vector,
    to_vector,
    as_value,
    mean_value,
)

from .core.parameterised import Parameterised, guard_pos

from .core.distribution import (
    Distribution,
    RegressionDistribution,
)

from .core.controllable import Controls, set_controls
from .core.estimator import Fittable, Data, Differentiable
from .core.regressor import Fittable as Regressable, UNSPECIFIED_REGRESSION


DEFAULT_LAMBDA = 1.0


#################################################################
# Poisson distribution


@set_controls(max_iters=0)
class PoissonDistribution(Parameterised, Distribution, Fittable):
    """
    Implements the Poisson probability distribution for a binary
    response variate, X.

    The sole parameter, _lambda, governs the probability that X=1.
    """

    # -----------------------
    # Parameterised interface

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_LAMBDA,)

    def __init__(self, _lambda: Value = DEFAULT_LAMBDA):
        """
        Initialises the Poisson distribution(s).

        Input:
            - _lambda (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(_lambda)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        _lambda = params[0]
        return not is_divergent(_lambda) and np.all(_lambda >= 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        _lambda = self.parameters()[0]
        return _lambda

    def variance(self) -> Value:
        _lambda = self.parameters()[0]
        return _lambda

    def log_prob(self, variate: VectorLike) -> Value:
        v_data = to_vector(variate)
        return as_value(self.compute_scores(self.parameters(), v_data))

    # ------------------
    # Fittable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        _lambda = mean_value(data.weights, data.variate)
        return (_lambda,)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        _lambda = guard_pos(params[0])
        return variate * np.log(_lambda) - _lambda - np.log(gamma(variate + 1))


#################################################################
# Poisson regression


class PoissonRegression(RegressionDistribution, Regressable, Differentiable):
    """
    Implements the Poisson conditional probability distribution
    for a binary response variate, X, as a linear regression of
    numerical covariate(s), Z.

    The link parameter is eta = log(lambda).
    """

    # --------------------------------
    # Parameterised interface

    @staticmethod
    def default_parameters() -> Values:
        return (UNSPECIFIED_REGRESSION,)

    def __init__(self, phi: Vector = UNSPECIFIED_REGRESSION):
        """
        Initialises the conditional Poisson distribution.

        Input:
            - phi (vector): The regression parameter value(s).
        """
        pdf = PoissonDistribution()
        super().__init__(pdf, phi)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        phi = params[0]
        return is_vector(phi) and not is_divergent(phi)

    # --------------------
    # Regression interface

    def invert_link(self, *link_params: Values) -> Values:
        _lambda = np.exp(link_params[0])
        return (_lambda,)

    # ---------------------
    # Estimator interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        ind = data.variate > 0
        if np.any(ind):
            eta = np.log(data.variate[ind])
            Z = data.covariates[ind, :]
            phi = solve(Z.T @ Z, Z.T @ eta)
        else:
            phi = np.zeros(data.covariates.shape[1])
        return (phi,)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        u_params = self.invert_link(*params)
        return self.distribution().compute_scores(u_params, variate)

    # ------------------------
    # Differentiable interface

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        # grad = dL/d eta NOT dL/d lambda
        _lambda = self.invert_link(*params)[0]
        return (variate - _lambda,)

    def compute_neg_hessian(self, params: Values, variate: Vector) -> Values2d:
        _lambda = self.invert_link(*params)[0]
        return ((_lambda,),)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    pd = PoissonDistribution()
    assert pd.parameters() == (DEFAULT_LAMBDA,)
    assert pd.mean() == DEFAULT_LAMBDA
    assert pd.variance() == DEFAULT_LAMBDA
    print("Passed default parameter tests!")

    # Test specified parameter
    LAMBDA = 0.123456
    pd = PoissonDistribution(LAMBDA)
    assert pd.parameters() == (LAMBDA,)
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
    pr = PoissonRegression()
    assert pr.parameters() == (UNSPECIFIED_REGRESSION,)
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

    pr = PoissonRegression()
    Zp1 = [(1, 1)] * len(Xp1)
    Zm1 = [(1, -1)] * len(Xm1)
    Z = np.concat((Zp1, Zm1))
    res = pr.fit(X, Z, score_tol=1e-12)
    assert res["converged"]
    assert np.abs(pr.regression_parameters()[0]) < 1e-6
    mu_p1 = pr.mean([1, 1])
    assert np.abs(mu_p1 - np.mean(Xp1)) < 1e-5
    mu_m1 = pr.mean([1, -1])
    assert np.abs(mu_m1 - np.mean(Xm1)) < 1e-5
    print("Passed fitting multiple regression tests!")
