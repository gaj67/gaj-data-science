"""
This module implements the gamma distribution.

The distributional parameters are alpha (shape) and beta (rate).

The link parameter is eta = ln(alpha/beta), which is not a natural
parameter.
"""

from typing import Optional

import numpy as np
from numpy.linalg import solve
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
    VectorLike,
    is_divergent,
    is_scalar,
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


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


#################################################################
# Beta distribution and fitter


class GammaDistribution(Parameterised, Distribution, Fittable, Differentiable):
    """
    Implements the gamma probability distribution for a non-negative
    response variate, X.
    """

    # -----------------------
    # Parameterised interface

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_ALPHA, DEFAULT_BETA)

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the beta distribution(s).

        Input:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 2:
            return False
        for param in params:
            if is_divergent(param) or np.any(param <= 0):
                return False
        return True

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta

    def variance(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta**2

    def log_prob(self, variate: VectorLike) -> Value:
        v_data = to_vector(variate)
        return as_value(self.compute_scores(self.parameters(), v_data))

    # ------------------
    # Fittable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        # Nonlinear approximation
        m = mean_value(data.weights, data.variate)
        m_ln = mean_value(data.weights, np.log(data.variate))
        s = np.log(m) - m_ln
        alpha = ((3 - s) + np.sqrt((3 - s) ** 2 + 24 * s)) / (12 * s)
        beta = alpha / m
        if self.is_valid_parameters(alpha, beta):
            return (alpha, beta)
        # Method of moments
        m2 = mean_value(data.weights, data.variate**2)
        v = m2 - m**2
        alpha = m**2 / v
        beta = m / v
        if self.is_valid_parameters(alpha, beta):
            return (alpha, beta)
        # Mean approximation
        alpha = DEFAULT_ALPHA
        beta = alpha / m
        return (alpha, beta)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        alpha, beta = params
        return gamma_dist.logpdf(variate, alpha, scale=1.0 / beta)

    # ------------------------
    # Differentiable interface

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        alpha, beta = params
        # d L / d alpha = ln X - E[ln X]
        mu_ln = digamma(alpha) - np.log(beta)
        # d L / d beta  = E[X] - X
        mu_x = alpha / beta
        return (np.log(variate) - mu_ln, mu_x - variate)

    def compute_neg_hessian(self, params: Values, variate: Vector) -> Values2d:
        alpha, beta = params
        # - d^2 L / d alpha^2 = Var[ln X]
        v_ln = polygamma(1, alpha)
        # - d^2 L / d beta^2 = Var[-X] = Var[X]
        v = alpha / beta**2
        # - d^2 L / d alpha d beta = Cov[ln X, -X] = -Cov[ln X, X]
        c = -1 / beta
        return ((v_ln, c), (c, v))


#################################################################
# Beta regression


DEFAULT_INDEPENDENT = 1


# @set_controls(use_external=True)
class GammaRegression(RegressionDistribution, Regressable, Differentiable):
    """
    Implements the gamma probability distribution for a non-negative
    response variate, X, as a linear regression of numerical covariate(s), Z.

    The link parameter is eta = ln(alpha/beta), and the independent
    parameter is psi = beta.
    """

    # --------------------------------
    # RegressionDistribution interface

    @staticmethod
    def default_parameters() -> Values:
        return (UNSPECIFIED_REGRESSION, DEFAULT_INDEPENDENT)

    def __init__(
        self, phi: Vector = UNSPECIFIED_REGRESSION, psi: Value = DEFAULT_INDEPENDENT
    ):
        """
        Initialises the conditional Beta distribution.

        Input:
            - phi (vector): The regression parameter value(s).
            - psi (float): The independent parameter value.
        """
        pdf = GammaDistribution()
        super().__init__(pdf, phi, psi)

    def is_valid_parameters(self, *params: Values) -> bool:
        if len(params) != 2:
            return False
        phi, psi = params
        if not is_vector(phi) or is_divergent(phi):
            return False
        return is_scalar(psi) and not is_divergent(psi) and psi > 0

    def invert_link(self, *link_params: Values) -> Values:
        eta, psi = link_params
        beta = psi
        alpha = psi * np.exp(eta)
        return (alpha, beta)

    # ---------------------
    # Regressable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        # phi = np.zeros(data.covariates.shape[1])
        eta = np.log(data.variate)
        Z = data.covariates
        phi = solve(Z.T @ Z, Z.T @ eta)
        return (phi, DEFAULT_INDEPENDENT)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        print("DEBUG[compute_scores]: params=", params)
        ab_params = self.invert_link(*params)
        print("DEBUG[compute_scores]: ab_params=", ab_params)
        print(
            "DEBUG[compute_scores]: scores=",
            self.distribution().compute_scores(ab_params, variate),
        )
        return self.distribution().compute_scores(ab_params, variate)

    # ------------------------
    # Differentiable interface

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        # params = (eta, psi)
        # grad = (dL/d eta, dL/d psi) NOT (dL/d alpha, dL/d beta)
        alpha, beta = ab_params = self.invert_link(*params)
        d_alpha, d_beta = self.distribution().compute_gradients(ab_params, variate)

        d_eta = alpha * d_alpha
        d_psi = d_beta + d_eta / beta
        print("DEBUG: d_eta=", d_eta, "d_psi=", d_psi)

        return (d_eta, d_psi)

    def compute_neg_hessian(self, params: Values, variate: Vector) -> Values2d:
        # Obtain dL / d{alpha, beta}
        alpha, beta = ab_params = self.invert_link(*params)
        d_alpha, d_beta = self.distribution().compute_gradients(ab_params, variate)

        # Obtain -d^2 L / d{alpha, beta}^2
        n_hess = self.distribution().compute_neg_hessian(ab_params, variate)
        d2_alpha, d2_ab = n_hess[0]
        d2_ab, d2_beta = n_hess[1]

        # Note: The gradient term vanishes under conditional expectation.
        # In practice, its presence makes little difference.
        d2_eta = alpha * (alpha * d2_alpha - d_alpha)
        d2_ep = alpha * d2_ab + d2_eta / beta
        a_on_b = alpha / beta
        d2_psi = d2_beta + a_on_b * (2 * d2_ab + a_on_b * d2_alpha)

        return ((d2_eta, d2_ep), (d2_ep, d2_psi))


###############################################################################


if __name__ == "__main__":
    # Test default parameter
    gd = GammaDistribution()
    assert gd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
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

    # Test fitting multiple observations
    for n in range(2, 11):
        X = np.random.randint(1, 100, n) * 0.1
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
    gr = GammaRegression()
    res = gr.fit(X, Z)
    assert res["converged"]
