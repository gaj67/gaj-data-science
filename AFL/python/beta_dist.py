"""
This module implements the Beta distribution.

The natural parameters are the distributional parameters, alpha and beta.
We choose the logit function for the link function, which means that
the link parameter, eta, is not one of the natural parameters.

For the regression model, eta depends on the regression parameters, phi.
In addition, we choose the shape parameter, alpha, to be independent of phi.
Hence, the other shape parameter, beta, depends on both alpha and eta.
"""
from typing import Optional, Tuple
from numpy import ndarray
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import loggamma, digamma, polygamma
from core_dist import ScalarPDF, Regressor, Fitting, Value, Values, Scalars
from stats_tools import weighted_mean, weighted_var, guard_prob


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@Regressor(max_iters=100, step_size=0.5)
# @Fitting(max_iters=1000, step_size=0.1)
class BetaDistribution(ScalarPDF):
    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the Beta distribution(s).

        Input:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

    def default_parameters(self) -> Scalars:
        return (DEFAULT_ALPHA, DEFAULT_BETA)

    def mean(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        alpha, beta = self.parameters()
        nu = alpha + beta
        mu = alpha / nu
        return mu * (1 - mu) / (nu + 1)

    def log_prob(self, X: Value) -> Value:
        alpha, beta = self.parameters()
        v1 = beta_dist.logpdf(X, alpha, beta)
        print("DEBUG: v1 =", v1)
        v2 = (
            (alpha - 1) * np.log(X)
            + (beta - 1) * np.log(1 - X)
            + loggamma(alpha + beta)
            - loggamma(alpha)
            - loggamma(beta)
        )
        print("DEBUG: v2 =", v2)
        return v2

    def internal_parameters(self) -> Values:
        alpha, beta = self.parameters()
        eta = np.log(alpha / beta)
        return (eta, alpha)

    def invert_parameters(self, eta: Value, psi: Value) -> Values:
        alpha = psi
        beta = alpha * np.exp(-eta)
        return (alpha, beta)

    def internal_variates(self, X: Value) -> Values:
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = np.log(X), np.log(1 - X)
        Y_eta = -beta * Y_beta
        Y_psi = Y_alpha + beta / alpha * Y_beta
        return (Y_eta, Y_psi)

    def internal_means(self) -> Values:
        alpha, beta = self.parameters()
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        mu_eta = -beta * mu_beta  # E[Y_eta]
        mu_psi = mu_alpha + beta / alpha * mu_beta  # E[Y_psi]
        return (mu_eta, mu_psi)

    def internal_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
        v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
        v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
        v_eta = beta**2 * v_beta  # Var[Y_eta]
        k = beta / alpha
        v_psi = v_alpha + k**2 * v_beta + 2 * k * cov_ab  # Var[Y_psi]
        cov_pe = -beta * (cov_ab + k * v_beta)  # Cov[Y_psi, Y_eta]
        return np.array([[v_eta, cov_pe], [cov_pe, v_psi]])

    def initialise_parameters(self, X: Value, W: Value, **kwargs: dict):
        mu = guard_prob(weighted_mean(W, X))
        sigma_sq = weighted_var(W, X)
        if sigma_sq > 0.01:
            nu = mu * (1 - mu) / sigma_sq
            alpha = mu * nu
            beta = nu - alpha
        else:
            beta = DEFAULT_BETA
            alpha = beta * mu / (1 - mu)
        self.set_parameters(alpha, beta)


###############################################################################


class BetaDistribution2(BetaDistribution):
    """
    Reimplements the Beta distribution with an alternative parameterisation
    of the link parameter and the independent parameters, as specified
    in notebooks/C_regression_models.ipynb#Beta-regression-(again).
    However, this implementation can no longer guarantee that the
    distributional parameters will remain positive.
    """

    def set_parameters(self, alpha, beta):
        # Check for divergence
        if isinstance(alpha, ndarray):
            assert all(alpha > 0)
        else:
            assert alpha > 0
        if isinstance(beta, ndarray):
            assert all(beta > 0)
        else:
            assert beta > 0
        super().set_parameters(alpha, beta)

    def internal_parameters(self) -> Values:
        alpha, beta = self.parameters()
        eta = alpha - beta
        psi = alpha + beta
        return (eta, psi)

    def invert_parameters(self, eta: Value, psi: Value) -> Values:
        alpha = (psi + eta) / 2
        beta = (psi - eta) / 2
        return (alpha, beta)

    def internal_variates(self, X: Value) -> Values:
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = np.log(X), np.log(1 - X)
        Y_eta = (Y_alpha - Y_beta) / 2
        Y_psi = (Y_alpha + Y_beta) / 2
        return (Y_eta, Y_psi)

    def internal_means(self) -> Values:
        alpha, beta = self.parameters()
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        mu_eta = (mu_alpha - mu_beta) / 2  # E[Y_eta]
        mu_psi = (mu_alpha + mu_beta) / 2  # E[Y_psi]
        return (mu_eta, mu_psi)

    def internal_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
        v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
        v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
        v_eta = (v_alpha + v_beta - 2 * cov_ab) / 4  # Var[Y_eta]
        v_psi = (v_alpha + v_beta + 2 * cov_ab) / 4  # Var[Y_psi]
        cov_pe = (v_alpha - v_beta) / 4  # Cov[Y_psi, Y_eta]
        return np.array([[v_eta, cov_pe], [cov_pe, v_psi]])


###############################################################################


if __name__ == "__main__":
    # Test default parameter
    bd = BetaDistribution()
    assert bd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = bd.mean()
    assert mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
    assert bd.variance() == mu * (1 - mu) / (1 + DEFAULT_ALPHA + DEFAULT_BETA)
    assert len(bd.internal_parameters()) == 2
    assert bd.internal_parameters()[0] == np.log(DEFAULT_ALPHA / DEFAULT_BETA)  # eta
    assert bd.internal_parameters()[1] == bd.parameters()[0]  # psi = alpha
    print("Passed default parameter tests!")

    # Test fitting multiple observations
    X = np.array([0.25, 0.75])
    bd = BetaDistribution()
    res = bd.fit(X)
    mu = bd.mean()
    assert np.abs(mu - 0.5) < 1e-3
    alpha, beta = bd.parameters()
    assert np.abs(alpha - beta) < 1e-3
    print("Passed simple fitting test!")

    # Test regression
    from core_dist import no_intercept, add_intercept

    # Test regression with intercept
    br = BetaDistribution().regressor([0, 0])
    Z = add_intercept([-1, +1])
    log_p, num_iters, tol = br.fit(X, Z)
    assert tol < 1e-6
    phi = br.regression_parameters()
    assert len(phi) == 2
    mu = br.mean(Z)
    assert all(np.abs(mu - X) < 1e-3)

    # Test alternative implementation
    br = BetaDistribution2().regressor()
    Z = add_intercept([-1, +1])
    log_p, num_iters, tol = br.fit(X, Z)
    assert tol < 1e-6
    phi = br.regression_parameters()
    assert len(phi) == 2
    mu = br.mean(Z)
    assert all(np.abs(mu - X) < 1e-3)
