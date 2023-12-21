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
import numpy as np
from numpy import ndarray
from scipy.stats import beta as beta_dist
from scipy.special import digamma, polygamma
from core_dist import ScalarPDF, RegressionPDF, Value, Values, Scalars


# Assume Jeffreys' prior:
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5


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
        return beta_dist.logpdf(X, alpha, beta)

    def link_parameters(self) -> Values:
        alpha, beta = self.parameters()
        eta = np.log(alpha / beta)
        return (eta, alpha)

    def link_inversion(self, eta: Value, *psi: Values) -> Values:
        alpha = psi[0] if len(psi) > 0 else self.parameters()[0]
        beta = alpha * np.exp(-eta)
        return (alpha, beta)

    def link_variates(self, X: Value) -> Values:
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = np.log(X), np.log(1 - X)
        Y_eta = -beta * Y_beta
        Y_psi = Y_alpha + beta / alpha * Y_beta
        return (Y_eta, Y_psi)

    def link_means(self) -> Values:
        alpha, beta = self.parameters()
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        mu_eta = -beta * mu_beta  # E[Y_eta]
        mu_psi = mu_alpha + beta / alpha * mu_beta  # E[Y_psi]
        return (mu_eta, mu_psi)

    def link_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
        v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
        v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
        v_eta = beta**2 * v_beta  # Var[Y_eta]
        k = beta / alpha
        v_psi = v_alpha + k**2 * v_beta + 2 * k * cov_ab  # Var[Y_psi]
        cov_pe = -beta * (cov_ab + k * v_beta)  # Cov[Y_psi, Y_eta]
        return np.array([[v_eta, cov_pe], [cov_pe, v_psi]])

    def regressor(self, phi: Optional[ndarray] = None) -> RegressionPDF:
        return BetaRegressionPDF(self, phi)


###############################################################################


# Reduce size of Newton-Raphson parameter update
DEFAULT_STEP_SIZE = 0.1


# Override the standard regressor in order to control parameter divergence
# when fitting the distribution to covariate data.
class BetaRegressionPDF(RegressionPDF):
    def fit(
        self,
        X: Value,
        Z: ndarray,
        W: Optional[Value] = None,
        max_iters: int = 100,
        min_tol: float = 1e-6,
        step_size: float = DEFAULT_STEP_SIZE,
    ) -> Tuple[float, int, float]:
        super().fit(X, Z, W, max_iters, min_tol, step_size)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BetaDistribution()
    assert bd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = bd.mean()
    assert mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
    assert bd.variance() == mu * (1 - mu) / (1 + DEFAULT_ALPHA + DEFAULT_BETA)
    assert len(bd.link_parameters()) == 2
    assert bd.link_parameters()[0] == np.log(DEFAULT_ALPHA / DEFAULT_BETA)  # eta
    assert bd.link_parameters()[1] == bd.parameters()[0]  # psi = alpha

    # Test fitting multiple observations
    X = np.array([0.25, 0.75])
    bd = BetaDistribution()
    res = bd.fit(X)
    mu = bd.mean()
    assert np.abs(mu - 0.5) < 1e-3
    alpha, beta = bd.parameters()
    assert np.abs(alpha - beta) < 1e-3

    # Test regression
    from core_dist import no_intercept, add_intercept

    # Test regression without intercept
    br = BetaDistribution().regressor()
    Z = no_intercept([-1, +1])
    res = br.fit(X, Z)
    phi = br.regression_parameters()
    assert len(phi) == 1
    mu = br.mean(Z)
    assert all(np.abs(mu - X) < 1e-3)
