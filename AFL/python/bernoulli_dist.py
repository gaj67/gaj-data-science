"""
This module implements the Bernoulli distribution.

The distributional parameter, theta, is also the mean, mu.
However, the natural parameter, eta, is the logit of theta (and thus mu).
Hence, we choose the logit as the link function, such that the link
parameter is identical to the natural parameter.

For the regression model, eta depends on the regression parameters, phi.
"""
import numpy as np
from numpy import ndarray
from core_dist import ScalarPDF, Value, Values, Scalars
from stats_tools import logistic, logit


DEFAULT_THETA = 0.5


class BernoulliDistribution(ScalarPDF):
    def __init__(self, theta: Value = DEFAULT_THETA):
        """
        Initialises the Bernoulli distribution(s).

        Input:
            - theta (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta)

    def default_parameters(self) -> Scalars:
        return (DEFAULT_THETA,)

    def mean(self) -> Value:
        theta = self.parameters()[0]
        return theta

    def variance(self) -> Value:
        theta = self.parameters()[0]
        return theta * (1 - theta)

    def log_prob(self, X: Value) -> Value:
        theta = self.parameters()[0]
        return X * np.log(theta) + (1 - X) * np.log(1 - theta)

    def link_parameters(self) -> Values:
        theta = self.parameters()[0]
        eta = logit(theta)
        return (eta,)

    def link_inversion(self, eta: Value, *psi: Values) -> Values:
        theta = logistic(eta)
        return (theta,)

    def link_variates(self, X: Value) -> Values:
        return (X,)

    def link_means(self) -> Values:
        return (self.mean(),)

    def link_variances(self) -> ndarray:
        return np.array([[self.variance()]])


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    bd = BernoulliDistribution()
    assert bd.parameters() == (DEFAULT_THETA,)
    assert bd.mean() == DEFAULT_THETA
    assert bd.variance() == DEFAULT_THETA * (1 - DEFAULT_THETA)
    assert len(bd.link_parameters()) == 1
    assert bd.link_parameters()[0] == 0.0
    X = np.array([0, 1, 1, 0])
    assert all(bd.link_variates(X)[0] == X)

    # Test specified parameter
    theta = 0.123456
    bd = BernoulliDistribution(theta)
    assert bd.parameters() == (theta,)
    assert bd.mean() == theta
    assert bd.variance() == theta * (1 - theta)

    # Test fitting 1 observation - be careful with 0 or 1!
    for x in [1e-3, 1, 0.5, 0.1]:
        bd = BernoulliDistribution()
        res = bd.fit(x)
        assert np.abs(bd.parameters()[0] - x) < 1e-6

    # Test fitting multiple observations
    for n in range(1, 11):
        X = np.random.randint(0, 2, n)
        bd = BernoulliDistribution()
        res = bd.fit(X)
        assert np.abs(bd.parameters()[0] - np.mean(X)) < 1e-6

    # Test regression
    from core_dist import RegressionPDF, no_intercept, add_intercept

    # Test regression without intercept
    br = RegressionPDF(BernoulliDistribution())
    X = np.array([1, 0])
    Z = no_intercept([1, -1])
    res = br.fit(X, Z)
    phi = br.regression_parameters()
    assert len(phi) == 1
    mu = br.mean(Z)
    for x, m in zip(X, mu):
        assert np.abs(m - x) < 1e-6

    # Test regression with intercept (or bias)
    br = RegressionPDF(BernoulliDistribution())
    X = np.array([1, 0, 1, 1, 0])
    Z = add_intercept([1, 1, -1, -1, -1])
    res = br.fit(X, Z)
    assert len(br.regression_parameters()) == 2
    mu = br.mean(add_intercept([1]))
    assert np.abs(mu - 0.5) < 1e-6
    mu = br.mean(add_intercept([-1]))
    assert np.abs(mu - 2 / 3) < 1e-6
