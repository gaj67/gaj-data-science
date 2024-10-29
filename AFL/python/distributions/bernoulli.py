"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
The link parameter is the natural parameter, eta = logit(theta).
"""

import numpy as np
from scipy.special import expit as logistic

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

from .core.parameterised import Parameterised, guard_prob

from .core.distribution import (
    Distribution,
    RegressionDistribution,
)

from .core.controllable import Controls, set_controls
from .core.estimator import Fittable, Data, Differentiable
from .core.regressor import Fittable as Regressable, UNSPECIFIED_REGRESSION


DEFAULT_THETA = 0.5


#################################################################
# Bernoulli distribution


@set_controls(max_iters=0)
class BernoulliDistribution(Parameterised, Distribution, Fittable):
    """
    Implements the Bernoulli probability distribution for a binary
    response variate, X.

    The sole parameter, theta, governs the probability that X=1.
    """

    # -----------------------
    # Parameterised interface

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_THETA,)

    def __init__(self, theta: Value = DEFAULT_THETA):
        """
        Initialises the Bernoulli distribution(s).

        Input:
            - theta (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        theta = params[0]
        return not is_divergent(theta) and np.all(theta >= 0) and np.all(theta <= 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        theta = self.parameters()[0]
        return theta

    def variance(self) -> Value:
        theta = self.parameters()[0]
        return theta * (1 - theta)

    def log_prob(self, variate: VectorLike) -> Value:
        v_data = to_vector(variate)
        return as_value(self.compute_scores(self.parameters(), v_data))

    # ------------------
    # Fittable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        theta = mean_value(data.weights, data.variate)
        return (theta,)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        theta = guard_prob(params[0])
        return variate * np.log(theta) + (1 - variate) * np.log(1 - theta)


#################################################################
# Bernoulli regression


class BernoulliRegression(RegressionDistribution, Regressable, Differentiable):
    """
    Implements the Bernoulli conditional probability distribution
    for a binary response variate, X, as a linear regression of
    numerical covariate(s), Z.

    The natural parameter is eta = logit(theta) with natural variate X.
    """

    # --------------------------------
    # RegressionDistribution interface

    @staticmethod
    def default_parameters() -> Values:
        return (UNSPECIFIED_REGRESSION,)

    def __init__(self, phi: Vector = UNSPECIFIED_REGRESSION):
        """
        Initialises the conditional Bernoulli distribution.

        Input:
            - phi (vector): The regression parameter value(s).
        """
        pdf = BernoulliDistribution()
        super().__init__(pdf, phi)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 1:
            return False
        phi = params[0]
        return is_vector(phi) and not is_divergent(phi)

    def invert_link(self, *link_params: Values) -> Values:
        theta = logistic(link_params[0])
        return (theta,)

    # ---------------------
    # Regressable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        phi = np.zeros(data.covariates.shape[1])
        return (phi,)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        u_params = self.invert_link(*params)
        return self.distribution().compute_scores(u_params, variate)

    # ------------------------
    # Differentiable interface

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        # grad = dL/d eta NOT dL/d theta
        theta = self.invert_link(*params)[0]
        return (variate - theta,)

    def compute_neg_hessian(self, params: Values, variate: Vector) -> Values2d:
        theta = self.invert_link(*params)[0]
        return ((theta * (1 - theta),),)


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
    assert br.parameters() == (UNSPECIFIED_REGRESSION,)
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
