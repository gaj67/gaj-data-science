"""
This module implements the Negative Binomial distribution.

The Polya form of the distribution may be parameterised by alpha and p, 
where alpha sepcifies the (possibly real-valued) 'number' of required 
events (e.g. successes), and p specifies the probability of such an event,
sampled at each trial over a sequence of independent Bernoulli trials.

When derived from a Poisson distribution with a Gamma(alpha, beta) prior,
we find that p = beta / (beta + 1) and beta = p / (1 - p).

The natural parameters are alpha and eta = log(1 - p).


We choose the XXX function for the link function, which means that
the link parameter, eta, is not one of the natural parameters.

For the regression model, eta depends on the regression parameters, phi.
In addition, we choose the shape parameter, alpha, to be independent of phi.
Hence, the other parameter, p (or beta), depends on both alpha and eta.
"""
from typing import Optional, Tuple
from numpy import ndarray
import numpy as np
from scipy.special import gamma, digamma, polygamma
from core_dist import ScalarPDF, RegressionPDF, Value, Values, Scalars


# Assume Exponential prior as a specialisation of Gamma(alpha, beta):
DEFAULT_ALPHA = 1
DEFAULT_PROB = 0.5


class NegBinomialDistribution(ScalarPDF):
    def __init__(self, alpha: Value = DEFAULT_ALPHA, p: Value = DEFAULT_PROB):
        """
        Initialises the Beta distribution(s).

        Input:
            - alpha (float or ndarray): The required number of stopping events.
            - p (float or ndarray): The probability of a stopping event.
        """
        super().__init__(alpha, p)

    def default_parameters(self) -> Scalars:
        return (DEFAULT_ALPHA, DEFAULT_PROB)

    def mean(self) -> Value:
        alpha, p = self.parameters()
        return alpha * (1 - p) / p

    def variance(self) -> Value:
        alpha, p = self.parameters()
        return alpha * (1 - p) / p**2

    def log_prob(self, X: Value) -> Value:
        alpha, p = self.parameters()
        return (
            np.log(gamma(alpha + X))
            - np.log(gamma(alpha))
            + X * np.log(1 - p)
            - np.log(gamma(X + 1))
            + alpha * np.log(p)
        )

    def link_parameters(self) -> Values:
        alpha, beta = self.parameters()
        XXXXX
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

    # Override fit() to control divergence
    def fit(
        self,
        X: Value,
        W: Optional[Value] = None,
        max_iters: int = 1000,
        min_tol: float = 1e-6,
        step_size: float = 0.1,
    ) -> Tuple[float, int, float]:
        # Enforce a single distribution, i.e. scalar parameter values
        if not self.is_scalar():
            self.reset_parameters()
        # Match moments to estimate better-than-default parameters
        if self.is_default():
            v = np.var(X)
            if v > 0.01:
                mu = np.mean(X)
                nu = mu * (1 - mu) / v
                alpha = mu * nu
                beta = nu - alpha
                self.set_parameters(alpha, beta)
        return super().fit(X, W, max_iters, min_tol, step_size)


###############################################################################


# Override the standard regressor in order to control parameter divergence
# when fitting the distribution to covariate data.
class BetaRegressionPDF(RegressionPDF):
    def fit(
        self,
        X: Value,
        Z: ndarray,
        W: Optional[Value] = None,
        max_iters: int = 10000,
        min_tol: float = 1e-6,
        step_size: float = 0.01,
    ) -> Tuple[float, int, float]:
        return super().fit(X, Z, W, max_iters, min_tol, step_size)


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

    def link_parameters(self) -> Values:
        alpha, beta = self.parameters()
        eta = alpha - beta
        psi = alpha + beta
        return (eta, psi)

    def link_inversion(self, eta: Value, *psi: Values) -> Values:
        if len(psi) > 0:
            psi = psi[0]
        else:
            alpha, beta = self.parameters()
            psi = alpha + beta
        alpha = (psi + eta) / 2
        beta = (psi - eta) / 2
        return (alpha, beta)

    def link_variates(self, X: Value) -> Values:
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = np.log(X), np.log(1 - X)
        Y_eta = (Y_alpha - Y_beta) / 2
        Y_psi = (Y_alpha + Y_beta) / 2
        return (Y_eta, Y_psi)

    def link_means(self) -> Values:
        alpha, beta = self.parameters()
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        mu_eta = (mu_alpha - mu_beta) / 2  # E[Y_eta]
        mu_psi = (mu_alpha + mu_beta) / 2  # E[Y_psi]
        return (mu_eta, mu_psi)

    def link_variances(self) -> ndarray:
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

    # Test regression with intercept
    br = BetaDistribution().regressor()
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
