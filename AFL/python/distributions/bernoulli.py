"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
"""

from typing import Optional

import numpy as np
from scipy.special import expit as logistic

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
)

from .core.distribution import Distribution, ConditionalDistribution, guard_prob
from .core.optimiser import Data, Controls, Results  # , fitting_controls
from .core.fitter import Fittable
from .core.regressor import Regressable, GradientRegressor, set_regressor


DEFAULT_THETA = 0.5


#################################################################
# Bernoulli distribution


class BernoulliDistribution(Distribution, Fittable):
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

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_THETA,)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        theta = params[0]
        return not is_divergent(theta) and np.all(theta >= 0) and np.all(theta <= 1)

    def mean(self) -> Value:
        theta = self.parameters()[0]
        return theta

    def variance(self) -> Value:
        theta = self.parameters()[0]
        return theta * (1 - theta)

    def log_prob(self, variate: VectorLike) -> Value:
        theta = guard_prob(self.parameters()[0])
        v_data = to_vector(variate)
        ln_p = v_data * np.log(theta) + (1 - v_data) * np.log(1 - theta)
        return ln_p if len(ln_p) > 1 else ln_p[0]

    def fit(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls,
    ) -> Results:
        data = self.to_data(variate, weights)
        theta = np.sum(data.weights * data.variate) / np.sum(data.weights)
        self.set_parameters(theta)
        score = np.sum(data.weights * self.log_prob(data.variate)) / np.sum(
            data.weights
        )
        return {
            "score": score,
            "num_iters": 0,
            "score_tol": 0.0,
            "converged": True,
        }


#################################################################
# Bernoulli regrression


DEFAULT_PHI = np.array([])


# @fitting_controls(step_size=0.1)
class BernoulliRegressor(GradientRegressor):
    """
    Implements optimisation of the regression parameters.

    Note: Scores and derivatives are with respeect to eta.
    """

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        phi = np.zeros(data.covariates.shape[1])
        return (phi,)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        eta = params[0]
        print("DEBUG[compute_scores]: eta =", eta)
        theta = logistic(eta)
        print("DEBUG[compute_scores]: theta =", theta)
        print("DEBUG[compute_scores]: variate =", variate)
        return variate * eta + np.log(1 - theta)

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        theta = logistic(params[0])
        return (variate - theta,)


@set_regressor(BernoulliRegressor)
class BernoulliRegression(ConditionalDistribution, Regressable):
    """
    Implements the Bernoulli conditional probability distribution
    for a binary response variate, X, as a linear regression of
    numerical covariate(s), Z.

    The natural parameter is eta = logit(theta) with natural variate X.
    """

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_PHI,)

    def __init__(self, phi: Vector = DEFAULT_PHI):
        """
        Initialises the conditional Bernoulli distribution.

        Input:
            - phi (vector): The regression parameter value(s).
        """
        super().__init__(phi)

    def is_valid_parameters(self, *params: Values) -> bool:
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
        theta = logistic(eta if len(eta) > 1 else eta[0])
        return (theta,)

    def mean(self, covariates: MatrixLike) -> Value:
        theta = self.inverse_parameters(covariates)[0]
        return theta

    def variance(self, covariates: MatrixLike) -> Value:
        theta = self.inverse_parameters(covariates)[0]
        return theta * (1 - theta)

    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        data = self.to_data(variate, None, covariates)
        theta = self.inverse_parameters(covariates)[0]
        log_probs = data.variate * np.log(theta) + (1 - data.variate) * np.log(
            1 - theta
        )
        return log_probs if len(log_probs) > 0 else log_probs[0]


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

    # ---------------------------------------------------

    # Test default parameter
    br = BernoulliRegression()
    assert br.parameters() == (DEFAULT_PHI,)
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
