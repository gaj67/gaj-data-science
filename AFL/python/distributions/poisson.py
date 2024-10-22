"""
This module implements the Poisson distribution.

The distributional parameter, lambda, is also the mean, mu.
The link parameter is the natural parameter, eta = log(lambda).
"""

from typing import Optional

import numpy as np
from scipy.special import gamma

import imptools

imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Vector,
    VectorLike,
    MatrixLike,
    is_divergent,
    to_matrix,
    is_vector,
    to_vector,
    mean_value,
)

from .core.distribution import Distribution, ConditionalDistribution, guard_pos
from .core.optimiser import Data, Controls, Results  # , fitting_controls
from .core.fitter import Fittable
from .core.regressor import Regressable, GradientRegressor, set_regressor


DEFAULT_LAMBDA = 1.0


#################################################################
# Poisson distribution


class PoissonDistribution(Distribution, Fittable):
    """
    Implements the Poisson probability distribution for a binary
    response variate, X.

    The sole parameter, _lambda, governs the probability that X=1.
    """

    def __init__(self, _lambda: Value = DEFAULT_LAMBDA):
        """
        Initialises the Poisson distribution(s).

        Input:
            - _lambda (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(_lambda)

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_LAMBDA,)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        _lambda = params[0]
        return not is_divergent(_lambda) and np.all(_lambda >= 0)

    def mean(self) -> Value:
        _lambda = self.parameters()[0]
        return _lambda

    def variance(self) -> Value:
        _lambda = self.parameters()[0]
        return _lambda

    def log_prob(self, variate: VectorLike) -> Value:
        _lambda = guard_pos(self.parameters()[0])
        v_data = to_vector(variate)
        ln_p = v_data * np.log(_lambda) - _lambda - np.log(gamma(v_data + 1))
        return ln_p if len(ln_p) > 1 else ln_p[0]

    def fit(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls,
    ) -> Results:
        data = self.to_data(variate, weights)
        _lambda = mean_value(data.weights, data.variate)
        self.set_parameters(_lambda)
        score = mean_value(data.weights, self.log_prob(data.variate))
        return {
            "score": score,
            "num_iters": 0,
            "score_tol": 0.0,
            "converged": True,
        }


#################################################################
# Poisson regrression


DEFAULT_PHI = np.array([])


# @fitting_controls(step_size=0.1)
class PoissonRegressor(GradientRegressor):
    """
    Implements optimisation of the regression parameters.

    Note: Scores and derivatives are with respeect to eta.
    """

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        n_cols = data.covariates.shape[1]
        print("DEBUG[estimate_parameters]: n_cols=", n_cols)
        scale = 1e-8 / n_cols
        phi = (np.random.uniform(size=n_cols) - 0.5) * scale
        phi =  np.zeros(n_cols)
        print("DEBUG[estimate_parameters]: phi =", phi)
        return (phi,)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        eta = params[0]
        print("DEBUG[compute_scores]: eta =", eta)
        _lambda = np.exp(eta)
        print("DEBUG[compute_scores]: _lambda =", _lambda)
        print("DEBUG[compute_scores]: variate =", variate)
        return variate * eta - _lambda - np.log(gamma(variate + 1))

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        _lambda = np.exp(params[0])
        return (variate - _lambda,)


@set_regressor(PoissonRegressor)
class PoissonRegression(ConditionalDistribution, Regressable):
    """
    Implements the Poisson conditional probability distribution
    for a binary response variate, X, as a linear regression of
    numerical covariate(s), Z.

    The link parameter is eta = log(lambda).
    """

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_PHI,)

    def __init__(self, phi: Vector = DEFAULT_PHI):
        """
        Initialises the conditional Poisson distribution.

        Input:
            - phi (vector): The regression parameter value(s).
        """
        super().__init__(phi)

    def is_valid_parameters(self, *params: Values) -> bool:
        print("DEBUG: params =", params)
        if len(params) != 1:
            return False
        phi = params[0]
        return is_vector(phi) and not is_divergent(phi)

    def inverse_parameters(self, covariates: MatrixLike) -> Values:
        phi = self.parameters()[0]
        if len(phi) == 0:
            raise ValueError("Uninitialised regression parameters!")
        covs = to_matrix(covariates, n_cols=len(phi))
        eta = covs @ phi
        _lambda = np.exp(eta if len(eta) > 1 else eta[0])
        return (_lambda,)

    def mean(self, covariates: MatrixLike) -> Value:
        _lambda = self.inverse_parameters(covariates)[0]
        return _lambda

    def variance(self, covariates: MatrixLike) -> Value:
        _lambda = self.inverse_parameters(covariates)[0]
        return _lambda

    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        data = self.to_data(variate, None, covariates)
        _lambda = self.inverse_parameters(covariates)[0]
        ln_p = data.variate * np.log(_lambda) - _lambda - np.log(gamma(data.variate + 1))
        return ln_p if len(ln_p) > 1 else ln_p[0]


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
    assert pr.parameters() == (DEFAULT_PHI,)
    print("Passed default regression parameter tests!")

    # Test fitting two groups of multiple observations
    Xp1 = [1, 2, 3, 1, 0, 0, 5, 4, 1, 0]
    Zp1 = [1] * len(Xp1)
    Xm1 = np.array(Xp1) / np.mean(Xp1)**2
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

    print("DEBUG: ***********************************")
    pr = PoissonRegression()
    Zp1 = [(1, 1)] * len(Xp1)
    Zm1 = [(1, -1)] * len(Xm1)
    Z = np.concat((Zp1, Zm1))
    res = pr.fit(X, Z, score_tol=1e-12)
    assert res["converged"]
    assert np.abs(pr.parameters()[0][0]) < 1e-6
    mu_p1 = pr.mean([1, 1])
    assert np.abs(mu_p1 - np.mean(Xp1)) < 1e-5
    mu_m1 = pr.mean([1, -1])
    assert np.abs(mu_m1 - np.mean(Xm1)) < 1e-5
    print("Passed fitting multiple regression tests!")
