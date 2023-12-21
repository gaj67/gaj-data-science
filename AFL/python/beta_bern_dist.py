"""
This module implements the Beta-Bernoulli distribution, derived by marginalising
a generative model that samples the Bernoulli parameter from a Beta distribution.

The distributional parameters are the shape parameters, alpha and beta.
However, the natural parameter, eta, is the logit of the mean, mu.
Hence, we choose the logit as the link function, such that the link
parameter is identical to the natural parameter.

For the regression model, eta depends on the regression parameters, phi.
In addition, we choose the shape parameter, alpha, to be independent of phi.
Hence, the other shape parameter, beta, depends on both alpha and eta.

Note that parameters alpha and beta induce variates Y_alpha and Y_beta,
for which the covariance matrix is singular. Strictly speaking, this means
that no Newton-Raphson update exists for the independent parameter, psi = alpha,
which should remain unestimated (i.e. constant). However, we instead use the
approximation that Y_psi = Y_alpha, i.e. that d beta/d alpha = 0.
"""
import numpy as np
from numpy import ndarray
from core_dist import ScalarPDF, Value, Values, Scalars


# Assume Jeffreys' prior:
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5


class BetaBernoulliDistribution(ScalarPDF):
    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the Beta-Bernoulli distribution(s).

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
        mu = self.mean()
        return mu * (1 - mu)

    def log_prob(self, X: Value) -> Value:
        alpha, beta = self.parameters()
        return X * np.log(alpha) + (1 - X) * np.log(beta) - np.log(alpha + beta)

    def link_parameters(self) -> Values:
        alpha, beta = self.parameters()
        eta = np.log(alpha / beta)
        return (eta, alpha)

    def link_inversion(self, eta: Value, *psi: Values) -> Values:
        alpha = psi[0] if len(psi) > 0 else self.parameters()[0]
        beta = alpha * np.exp(-eta)
        return (alpha, beta)

    # XXX Approximate Y_psi by neglecting the dependence of beta on alpha

    def link_variates(self, X: Value) -> Values:
        alpha = self.parameters()[0]
        return (X, X / alpha)

    def link_means(self) -> Values:
        alpha = self.parameters()[0]
        mu = self.mean()
        return (mu, mu / alpha)

    # XXX Also neglect the correlation between Y_eta and Y_psi

    def link_variances(self) -> ndarray:
        alpha = self.parameters()[0]
        v = self.variance()
        cov = 0.0 if not isinstance(v, ndarray) else np.zeros(len(v))
        return np.array([[v, cov], [cov, v / alpha**2]])


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BetaBernoulliDistribution()
    assert bd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = bd.mean()
    assert mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
    assert bd.variance() == mu * (1 - mu)
    assert len(bd.link_parameters()) == 2
    assert bd.link_parameters()[1] == bd.parameters()[0]  # alpha
    assert bd.link_parameters()[0] == np.log(DEFAULT_ALPHA / DEFAULT_BETA)
    X = np.array([0, 1, 1, 0])
    assert all(bd.link_variates(X)[0] == X)
    assert all(bd.link_variates(X)[1] == X / DEFAULT_ALPHA)

    # Test fitting multiple observations
    X = np.array([1, 1, 0, 1])
    bd = BetaBernoulliDistribution()
    res = bd.fit(X)
    assert np.abs(bd.mean() - np.mean(X)) < 1e-6
    alpha, beta = bd.parameters()
    assert np.abs(alpha / beta - 3) < 1e-5

    # Test regression
    from core_dist import no_intercept, add_intercept

    # Test regression without intercept
    br = BetaBernoulliDistribution().regressor()
    X = np.array([1, 0])
    Z = no_intercept([1, -1])
    res = br.fit(X, Z)
    phi = br.regression_parameters()
    assert len(phi) == 1
    mu = br.mean(Z)
    for x, m in zip(X, mu):
        assert np.abs(m - x) < 1e-6

    # Test regression with intercept (or bias)
    br = BetaBernoulliDistribution().regressor()
    X = np.array([1, 0, 1, 1, 0])
    Z = add_intercept([1, 1, -1, -1, -1])
    res = br.fit(X, Z)
    assert len(br.regression_parameters()) == 2
    mu = br.mean(add_intercept([1]))
    assert np.abs(mu - 0.5) < 1e-6
    mu = br.mean(add_intercept([-1]))
    assert np.abs(mu - 2 / 3) < 1e-6
