"""
This module implements the Poisson distribution.

The distributional parameter, lambda, is also the mean, mu, and the variance.
However, the natural parameter, eta, is the log of lambda (and thus mu).
Hence, we choose the log as the link function, such that the link
parameter is identical to the natural parameter.

For the regression model, eta depends on the regression parameters, phi.
"""

import numpy as np
from scipy.special import gamma

from scalar_pdf import (
    ScalarPDF,
    Value,
    Values,
    Values2D,
    check_transformations,
)

from stats_tools import weighted_mean, guard_pos


DEFAULT_LAMBDA = 1


class PoissonPDF(ScalarPDF):
    """
    Implements the Poisson probability distribution for a non-negative,
    integer response variate, X.
    """

    def __init__(self, lam: Value = DEFAULT_LAMBDA):
        """
        Initialises the Poisson distribution(s).

        Input:
            - lam (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(lam)

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_LAMBDA,)

    def mean(self) -> Value:
        lam = self.parameters()[0]
        return lam

    def variance(self) -> Value:
        lam = self.parameters()[0]
        return lam

    def log_prob(self, data: Value) -> Value:
        lam = guard_pos(self.parameters()[0])
        return data * np.log(lam) - lam - np.log(gamma(data + 1))

    def _internal_parameters(self, *theta: Values) -> Values:
        lam = guard_pos(theta[0])
        eta = np.log(lam)
        return (eta,)

    def _distributional_parameters(self, *psi: Values) -> Values:
        eta = psi[0]
        lam = np.exp(eta)
        return (lam,)

    def _estimate_parameters(
        self, data: Value, weights: Value, **kwargs: dict
    ) -> Values:
        lam = weighted_mean(weights, data)
        return (lam,)

    def _internal_gradient(self, data: Value) -> Values:
        # d L / d eta = X - E[X]
        mu = self.mean()
        return (data - mu,)

    def _internal_neg_hessian(self, data: Value) -> Values2D:
        # - d^2 L / d eta^2 = Var[X]
        v = self.variance()
        return ((v,),)


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    pd = PoissonPDF()
    assert pd.parameters() == (DEFAULT_LAMBDA,)
    assert pd.mean() == DEFAULT_LAMBDA
    assert pd.variance() == DEFAULT_LAMBDA
    print("Passed default parameter tests!")

    # Test specified parameter
    LAMBDA = 0.123456
    pd = PoissonPDF(LAMBDA)
    assert pd.parameters() == (LAMBDA,)
    assert pd.mean() == LAMBDA
    assert pd.variance() == LAMBDA
    print("Passed specified parameter tests!")

    assert check_transformations(pd)
    print("Passed parameter transformations tests!")

    # Test fitting 1 observation
    for value in [0, 1, 10, 100]:
        pd = PoissonPDF()
        res = pd.fit(value)
        assert np.abs(pd.mean() - value) < 1e-6
    print("Passed fitting 1 observation tests!")

    # Test fitting multiple observations
    for n in range(2, 11):
        values = np.random.randint(0, 100, n)
        pd = PoissonPDF()
        res = pd.fit(values)
        assert np.abs(pd.mean() - np.mean(values)) < 1e-6
    print("Passed fitting multiple observations tests!")
