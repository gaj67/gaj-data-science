"""
This module implements the gamma distribution.

The distributional parameters are alpha (shape) and beta (rate).

The link parameter is taken to be eta = log(alpha/beta) = log(mu),
which is not a natural parameter. The independent parameter is chosen
as psi = beta.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, polygamma

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

from .core.distribution import StandardDistribution, set_link_model
from .core.link_models import LogRatioLink21b
from .core.optimiser import Data, Controls


#################################################################
# Gamma distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_link_model(LogRatioLink21b)
class GammaDistribution(StandardDistribution):
    """
    Implements the gamma probability distribution for a non-negative
    response variate, X.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the gamma distribution(s).

        Input:
            - alpha (scalar or vector): The shape parameter value(s).
            - beta (scalar or vector): The rate parameter value(s).
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
        return alpha / beta

    def variance(self) -> Value:
        alpha, beta = self.get_parameters()
        return alpha / beta**2

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        # Nonlinear approximation
        m = mean_value(data.weights, data.variate)
        m_ln = mean_value(data.weights, np.log(data.variate))
        s = np.log(m) - m_ln
        alpha = ((3 - s) + np.sqrt((3 - s) ** 2 + 24 * s)) / (12 * s)
        beta = alpha / m
        if self.check_parameters(alpha, beta):
            print("DEBUG[nonlinear]: alpha=", alpha, "beta=", beta)
            return (alpha, beta)
        # Method of moments
        m2 = mean_value(data.weights, data.variate**2)
        v = m2 - m**2
        alpha = m**2 / v
        beta = m / v
        if self.check_parameters(alpha, beta):
            print("DEBUG[moments]: alpha=", alpha, "beta=", beta)
            return (alpha, beta)
        # Mean approximation
        alpha = DEFAULT_ALPHA
        beta = alpha / m
        print("DEBUG[backoff]: alpha=", alpha, "beta=", beta)
        return (alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self.get_parameters()
        return gamma_dist.logpdf(variate, alpha, scale=1.0 / beta)

    def compute_gradients(self, variate: Vector) -> Values:
        alpha, beta = self.get_parameters()
        # d L / d alpha = ln X - E[ln X]
        mu_ln = digamma(alpha) - np.log(beta)
        # d L / d beta  = E[X] - X
        mu_x = alpha / beta
        return (np.log(variate) - mu_ln, mu_x - variate)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        alpha, beta = self.get_parameters()
        # - d^2 L / d alpha^2 = Var[ln X]
        v_ln = polygamma(1, alpha)
        # - d^2 L / d beta^2 = Var[-X] = Var[X]
        v_x = alpha / beta**2
        # - d^2 L / d alpha d beta = Cov[ln X, -X] = -Cov[ln X, X]
        c_ab = -1 / beta
        return ((v_ln, c_ab), (c_ab, v_x))


###############################################################################


if __name__ == "__main__":
    # Test default parameter
    gd = GammaDistribution()
    assert gd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    assert gd.mean() == DEFAULT_ALPHA / DEFAULT_BETA
    assert gd.variance() == DEFAULT_ALPHA / DEFAULT_BETA**2
    print("Passed default parameter tests!")

    # Test fitting two observations
    X = np.array([1.0, 10])
    gd = GammaDistribution()
    res = gd.fit(X)
    print("DEBUG: gd.mean=", gd.mean(), "np.mean=", np.mean(X))
    assert np.abs(gd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting simple observations test!")

    # Test fitting multiple observations - possible divergence
    X = [3.4, 9.4, 9.6, 8.8, 8.4, 0.2, 4.3]
    gd = GammaDistribution()
    res = gd.fit(X)
    assert np.abs(gd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting problematic observations test!")

    X = [4.3, 3.8]
    gd = GammaDistribution()
    res = gd.fit(X)
    assert np.abs(gd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting second problematic observations test!")

    # Test fitting multiple observations
    for n in range(2, 11):
        X = np.random.randint(1, 100, n) * 0.1
        print("DEBUG[gammma]: X=", X)
        gd = GammaDistribution()
        res = gd.fit(X)
        assert np.abs(gd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")

    # Test two-group regression
    Xm1 = [3.4, 0.2, 4.3]
    Xp1 = [9.4, 9.6, 8.8, 8.4]
    Zm1 = [(1, -1)] * len(Xm1)
    Zp1 = [(1, 1)] * len(Xp1)
    X = np.hstack((Xm1, Xp1))
    Z = np.vstack((Zm1, Zp1))
    gr = GammaDistribution().regressor()
    res = gr.fit(X, Z)
    assert res["converged"]

    # Test regression with only bias
    Z0 = [1] * len(X)
    gr = GammaDistribution().regressor()
    res = gr.fit(X, Z0)
    wvec, b = gr.get_parameters()
    w = wvec[0]
    gd = GammaDistribution()
    res = gd.fit(X)
    a0, b0 = gd.get_parameters()
    w0 = np.log(a0 / b0)
    assert np.abs(w - w0) < 1e-15
    assert np.abs(b - b0) < 1e-15
    a1, b1 = gr.link_model().invert_transform(w, b)
    assert np.abs(a1 - a0) < 1e-15
    assert np.abs(b1 - b0) < 1e-15
    w2, b2 = gr.link_model().apply_transform(a1, b1)
    assert np.abs(w - w2) < 1e-15
    assert np.abs(b - b2) < 1e-15
    print("Passed bias-only regression tests!")
