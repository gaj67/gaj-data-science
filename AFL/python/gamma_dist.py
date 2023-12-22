"""
This module implements the Gamma distribution.

The natural parameters are the distributional parameters, alpha and beta.
We choose the log function for the link function, which means that
the link parameter, eta, is not one of the natural parameters.

For the regression model, eta depends on the regression parameters, phi.
In addition, we choose the rate parameter, beta, to be independent of phi.
Hence, the shape parameter, alpha, depends on both beta and eta.
"""
from typing import Optional, Tuple
import numpy as np
from numpy import ndarray
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, polygamma
from core_dist import ScalarPDF, RegressionPDF, Value, Values, Scalars


# Assume Exponential prior:
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0


class GammaDistribution(ScalarPDF):
    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the Gamma distribution(s).

        Inputs:
            - alpha (float or ndarray): The shape parameter value(s).
            - beta (float or ndarray): The rate parameter value(s).
        """
        super().__init__(alpha, beta)

    def default_parameters(self) -> Scalars:
        return (DEFAULT_BETA, DEFAULT_BETA)

    def mean(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta

    def variance(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta**2

    def log_prob(self, X: Value) -> Value:
        alpha, beta = self.parameters()
        return gamma_dist.logpdf(X, alpha, scale=1.0 / beta)

    def link_parameters(self) -> Values:
        alpha, beta = self.parameters()
        eta = np.log(alpha / beta)
        return (eta, beta)

    def link_inversion(self, eta: Value, *psi: Values) -> Values:
        beta = psi[0] if len(psi) > 0 else self.parameters()[1]
        alpha = beta * np.exp(eta)
        return (alpha, beta)

    def link_variates(self, X: Value) -> Values:
        alpha, beta = self.parameters()
        Y_eta = alpha * np.log(X)
        # Y_psi = alpha / beta * np.log(X) - X
        Y_psi = Y_eta / beta - X
        return (Y_eta, Y_psi)

    def link_means(self) -> Values:
        alpha, beta = self.parameters()
        mu_X = self.mean()
        mu_lnX = digamma(alpha) - np.log(beta)
        mu_eta = alpha * mu_lnX
        # mu_psi = -mu_X + alpha / beta * mu_lnX
        mu_psi = mu_X * (mu_lnX - 1)
        return (mu_eta, mu_psi)

    def link_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        v_X = self.variance()
        v_lnX = polygamma(1, alpha)
        # c_XlnX = 1.0 / beta
        v_eta = alpha**2 * v_lnX
        # k = alpha / beta
        # v_psi = v_X + k**2 * v_lnX - 2 * k * c_XlnX
        v_psi = v_X + (v_eta - 2 * alpha) / beta**2
        # cov = -alpha * c_XlnX + alpha * k * v_lnX
        cov = (v_eta - alpha) / beta
        return np.array([[v_eta, cov], [cov, v_psi]])

    # Override fit() to control divergence
    def fit(
        self,
        X: Value,
        W: Optional[Value] = None,
        max_iters: int = 1000,
        min_tol: float = 1e-6,
        step_size: float = 0.01,
    ) -> Tuple[float, int, float]:
        # Enforce a single distribution, i.e. scalar parameter values
        if not self.is_scalar():
            self.reset_parameters()
        # Match moments to estimate better-than-default parameters
        if self.is_default():
            v = np.var(X)
            if v > 0.01:
                mu = np.mean(X)
                beta = mu / v
                alpha = beta * mu
                self.set_parameters(alpha, beta)
        # Dynamically reduce gradient step-size if necessary
        n = int(np.log10(np.max(X)))
        _step_size = 0.1 ** (n + 1)
        step_size = min(step_size, _step_size)
        # Dynamically increase number of iterations if necessary
        _max_iters = 10 ** (n + 2)
        max_iters = max(max_iters, _max_iters)
        return super().fit(X, W, max_iters, min_tol, step_size)

    # Override conditional PDF to control divergence
    def regressor(self, phi: Optional[ndarray] = None) -> RegressionPDF:
        return GammaRegressionPDF(self, phi)


###############################################################################


# Override the standard regressor in order to control parameter divergence
# when fitting the distribution to covariate data.
class GammaRegressionPDF(RegressionPDF):
    def fit(
        self,
        X: Value,
        Z: ndarray,
        W: Optional[Value] = None,
        max_iters: int = 1000,
        min_tol: float = 1e-6,
        step_size: float = 0.01,
    ) -> Tuple[float, int, float]:
        # Dynamically reduce gradient step-size if necessary
        n = int(np.log10(np.max(X)))
        _step_size = 0.1 ** (n + 1)
        step_size = min(step_size, _step_size)
        # Dynamically increase number of iterations if necessary
        _max_iters = 10 ** (n + 2)
        max_iters = max(max_iters, _max_iters)
        return super().fit(X, Z, W, max_iters, min_tol, step_size)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    gd = GammaDistribution()
    assert gd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = gd.mean()
    assert mu == DEFAULT_ALPHA / DEFAULT_BETA
    assert gd.variance() == DEFAULT_ALPHA / DEFAULT_BETA**2
    assert len(gd.link_parameters()) == 2
    assert gd.link_parameters()[0] == np.log(DEFAULT_ALPHA / DEFAULT_BETA)  # eta
    assert gd.link_parameters()[1] == gd.parameters()[1]  # psi = beta

    # Test fitting multiple observations
    X = np.array([1.0, 10])
    gd = GammaDistribution()
    res = gd.fit(X)
    mu = gd.mean()
    alpha, beta = gd.parameters()
    assert not np.isnan(alpha)
    assert not np.isnan(beta)

    # Test regression
    from core_dist import no_intercept, add_intercept

    # Test regression with intercept
    gr = GammaDistribution().regressor()
    assert len(gr.independent_parameters()) == 1
    X1 = np.array([1, 2, 3])
    Z1 = -1
    X2 = np.array([100, 200, 300])
    Z2 = 1
    X = np.concatenate((X1, X2))
    Z = add_intercept([Z1] * len(X1) + [Z2] * len(X2))
    res = gr.fit(X, Z)
    phi = gr.regression_parameters()
    assert len(phi) == 2
    assert not np.isnan(phi[0])
    assert not np.isnan(phi[1])
    mu1 = gr.mean(add_intercept([Z1]))
    mu2 = gr.mean(add_intercept([Z2]))
    assert mu2 > mu1
    assert np.abs(np.log10(mu1 / np.mean(X1))) < 1
    assert np.abs(np.log10(mu2 / np.mean(X2))) < 1
