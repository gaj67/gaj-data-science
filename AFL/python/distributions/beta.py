"""
This module implements the beta distribution.

The distributional parameters are alpha and beta.
The link parameter is taken to be eta = ln(alpha/beta),
which is not a natural parameter.
"""

from typing import Optional

import numpy as np
from numpy.linalg import solve
from scipy.stats import beta as beta_dist
from scipy.special import digamma, polygamma, logit

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

from .core.parameterised import Parameterised, guard_prob

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


class BetaDistribution(Parameterised, Distribution, Fittable, Differentiable):
    """
    Implements the beta probability distribution for a proportional
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
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        alpha, beta = self.parameters()
        nu = alpha + beta
        mu = alpha / nu
        return mu * (1 - mu) / (nu + 1)

    def log_prob(self, variate: VectorLike) -> Value:
        v_data = to_vector(variate)
        return as_value(self.compute_scores(self.parameters(), v_data))

    # ------------------
    # Fittable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        # Backoff estimate based only on mean
        mu_X = mean_value(data.weights, data.variate)
        beta = DEFAULT_BETA
        alpha = beta * mu_X / (1 - mu_X)
        # Estimate based on variance (method of moments)
        mu_Xsq = mean_value(data.weights, data.variate**2)
        var_X = mu_Xsq - mu_X**2
        if var_X > 0:
            nu = mu_X * (1 - mu_X) / var_X - 1
            if nu > 0:
                alpha = nu * mu_X
                beta = nu - alpha
        return (alpha, beta)

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        return beta_dist.logpdf(variate, *params)

    # ------------------------
    # Differentiable interface

    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        # grad = (dL/d alpha, dL/d beta)
        alpha, beta = params
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        return (np.log(variate) - mu_alpha, np.log(1 - variate) - mu_beta)

    def compute_neg_hessian(self, params: Values, variate: Vector) -> Values2d:
        alpha, beta = params
        cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
        v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
        v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
        return ((v_alpha, cov_ab), (cov_ab, v_beta))


#################################################################
# Beta regression


DEFAULT_INDEPENDENT = 1


#@set_controls(use_external=True)
class BetaRegression(RegressionDistribution, Regressable, Differentiable):
    """
    Implements the beta probability distribution for a proportional
    response variate, X, as a linear regression of numerical covariate(s), Z.

    The link parameter is eta = ln(alpha/beta), and the independent
    parameter is psi = alpha.
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
        pdf = BetaDistribution()
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
        alpha = psi
        beta = psi * np.exp(-eta)
        return (alpha, beta)

    # ---------------------
    # Regressable interface

    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        # phi = np.zeros(data.covariates.shape[1])
        eta = logit(data.variate)
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

        d_eta = -beta * d_beta
        d_psi = d_alpha - d_eta / alpha

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
        # ??? In practice, its presence makes little difference.
        d2_eta = beta * (beta * d2_beta - d_beta)
        d2_ep = -beta * d2_ab - d2_eta / alpha
        b_on_a = beta / alpha
        d2_psi = d2_alpha + b_on_a * (2 * d2_ab + b_on_a * d2_beta)

        print("DEBUG: d2_eta=", d2_eta, "d2_psi=", d2_psi, "d2_ep=", d2_ep)
        return ((d2_eta, d2_ep), (d2_ep, d2_psi))


###############################################################################


if __name__ == "__main__":
    # Test default parameter
    bd = BetaDistribution()
    assert bd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = bd.mean()
    assert mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
    assert bd.variance() == mu * (1 - mu) / (1 + DEFAULT_ALPHA + DEFAULT_BETA)
    print("Passed default parameter tests!")

    # Test fitting two observations
    X = np.array([0.25, 0.75])
    bd = BetaDistribution()
    res = bd.fit(X)
    mu = bd.mean()
    assert np.abs(mu - 0.5) < 1e-6
    alpha, beta = bd.parameters()
    assert np.abs(alpha - beta) < 1e-6
    print("Passed simple fitting test!")

    # Test same data via regression - means are complementary
    br = BetaRegression()
    Z = [-1, 1]
    res = br.fit(X, Z)

    # Test regression on two groups - means are NOT complementary
    Xm1 = [0.1, 0.2, 0.3]  # mean 0.2
    Xp1 = [0.6, 0.7, 0.8]  # mean 0.7
    Zm1 = [(1, -1)] * len(Xm1)
    Zp1 = [(1, 1)] * len(Xp1)
