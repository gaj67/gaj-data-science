"""
This module implements the Poisson distribution.

The distributional parameter, lambda, is also the mean, mu, and the variance.
However, the natural parameter, eta, is the log of lambda (and thus mu).
Hence, we choose the log as the link function, such that the link
parameter is identical to the natural parameter.

For the regression model, eta depends on the regression parameters, phi.
"""
from typing import Optional, Tuple
from numpy import ndarray
import numpy as np
from core_dist import ScalarPDF, Regressor, Value, Values, Scalars
from scipy.special import gamma
from stats_tools import weighted_mean, guard_pos


DEFAULT_LAMBDA = 1


@Regressor
class PoissonDistribution(ScalarPDF):
    def __init__(self, lam: Value = DEFAULT_LAMBDA):
        """
        Initialises the Poisson distribution(s).

        Input:
            - lam (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(lam)

    def default_parameters(self) -> Scalars:
        return (DEFAULT_LAMBDA,)

    def mean(self) -> Value:
        lam = self.parameters()[0]
        return lam

    def variance(self) -> Value:
        lam = self.parameters()[0]
        return lam

    def log_prob(self, X: Value) -> Value:
        lam = guard_pos(self.parameters()[0])
        return X * np.log(lam) - lam - np.log(gamma(X + 1))

    def internal_parameters(self) -> Values:
        lam = guard_pos(self.parameters()[0])
        eta = np.log(lam)
        return (eta,)

    def invert_parameters(self, eta: Value) -> Values:
        lam = np.exp(eta)
        return (lam,)

    def internal_variates(self, X: Value) -> Values:
        return (X,)

    def internal_means(self) -> Values:
        return (self.mean(),)

    def internal_variances(self) -> ndarray:
        return np.array([[self.variance()]])

    def initialise_parameters(self, X: Value, W: Value, **kwargs: dict):
        lam = weighted_mean(W, X)
        self.set_parameters(lam)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    pd = PoissonDistribution()
    assert pd.parameters() == (DEFAULT_LAMBDA,)
    assert pd.mean() == DEFAULT_LAMBDA
    assert pd.variance() == DEFAULT_LAMBDA
    assert len(pd.internal_parameters()) == 1
    assert pd.internal_parameters()[0] == 0.0
    X = np.arange(5)
    assert all(pd.internal_variates(X)[0] == X)
    print("Passed default parameter tests!")

    # Test specified parameter
    lam = 0.123456
    pd = PoissonDistribution(lam)
    assert pd.parameters() == (lam,)
    assert pd.mean() == lam
    assert pd.variance() == lam
    print("Passed specified parameter tests!")

    # Test fitting 1 observation
    for x in [0, 1, 10, 100]:
        pd = PoissonDistribution()
        res = pd.fit(x)
        assert np.abs(pd.parameters()[0] - x) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    for n in range(2, 11):
        X = np.random.randint(0, 100, n)
        pd = PoissonDistribution()
        res = pd.fit(X)
        assert np.abs(pd.parameters()[0] - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")

    # Test regression
    from core_dist import no_intercept, add_intercept

    # Test regression with intercept (or bias)
    pr = PoissonDistribution().regressor()
    X = np.array([10, 20, 1, 2, 3])
    Z = add_intercept([1, 1, -1, -1, -1])
    res = pr.fit(X, Z)
    assert len(pr.regression_parameters()) == 2
    mu = pr.mean(add_intercept([1]))
    assert np.abs(mu - 15) < 1e-6
    mu = pr.mean(add_intercept([-1]))
    assert np.abs(mu - 2) < 1e-6
    print("Passed regression fitting tests!")
