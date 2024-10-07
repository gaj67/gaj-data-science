"""
This module implements the Gamma distribution.

The natural parameters are the distributional parameters, alpha and beta.
We choose the log function for the link function, which means that
the link parameter, eta, is not one of the natural parameters.

For the regression model, eta depends on the regression parameters, phi.
In addition, we choose the rate parameter, beta, to be independent of phi.
Hence, the shape parameter, alpha, depends on both beta and eta.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, polygamma

from scalar_pdf import ScalarPDF, Value, Values, Values2D, is_divergent
from stats_tools import weighted_mean, weighted_var


# Assume Exponential prior:
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0


class GammaPDF(ScalarPDF):
    """
    Implements the gamma probability distribution for a non-negative,
    real-valued response variate, X.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the gamma distribution(s).

        Inputs:
            - alpha (float or ndarray): The shape parameter value(s).
            - beta (float or ndarray): The rate parameter value(s).
        """
        super().__init__(alpha, beta)

    @staticmethod
    def default_parameters() -> Values:
        return (DEFAULT_BETA, DEFAULT_BETA)

    @staticmethod
    def is_valid_parameters(*params: Values) -> bool:
        if len(params) != 2:
            return False
        for value in params:
            if is_divergent(value) or np.any(value <= 0):
                return False
        return True

    def mean(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta

    def variance(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta**2

    def log_prob(self, data: Value) -> Value:
        alpha, beta = self.parameters()
        return gamma_dist.logpdf(data, alpha, scale=1.0 / beta)

    def _estimate_parameters(
        self, data: Value, weights: Value, **kwargs: dict
    ) -> Values:
        # Nonlinear approximation
        m = weighted_mean(weights, data)
        m_ln = weighted_mean(weights, np.log(data))
        s = np.log(m) - m_ln
        alpha = ((3 - s) + np.sqrt((3 - s) ** 2 + 24 * s)) / (12 * s)
        beta = alpha / m
        if self.is_valid_parameters(alpha, beta):
            return (alpha, beta)
        ## Method of moments
        v = weighted_var(weights, data)
        alpha = m**2 / v
        beta = m / v
        if self.is_valid_parameters(alpha, beta):
            return (alpha, beta)
        # Mean approximation
        alpha = DEFAULT_ALPHA
        beta = alpha / m
        return (alpha, beta)

    def _internal_gradient(self, data: Value) -> Values:
        alpha, beta = self.parameters()
        # d L / d alpha = ln X - E[ln X]
        mu_ln = digamma(alpha) - np.log(beta)
        # d L / d beta  = E[X] - X
        mu_x = alpha / beta
        return (np.log(data) - mu_ln, mu_x - data)

    def _internal_neg_hessian(self, data: Value) -> Values2D:
        alpha, beta = self.parameters()
        # - d^2 L / d alpha^2 = Var[ln X]
        v_ln = polygamma(1, alpha)
        # - d^2 L / d beta^2 = Var[-X] = Var[X]
        v = alpha / beta**2
        # - d^2 L / d alpha d beta = Cov[ln X, -X] = -Cov[ln X, X]
        c = -1 / beta
        return ((v_ln, c), (c, v))


###############################################################################

if __name__ == "__main__":
    # Test default parameter
    gd = GammaPDF()
    assert gd.parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    mu = gd.mean()
    assert mu == DEFAULT_ALPHA / DEFAULT_BETA
    assert gd.variance() == DEFAULT_ALPHA / DEFAULT_BETA**2
    print("Passed default parameter tests!")

    # Test fitting two observations
    values = np.array([1.0, 10])
    gd = GammaPDF()
    res = gd.fit(values)
    mu = gd.mean()
    assert np.abs(gd.mean() - np.mean(values)) < 1e-6
    print("Passed fitting simple observations test!")

    # Test fitting multiple observations - possible divergence
    values = [3.4, 9.4, 9.6, 8.8, 8.4, 0.2, 4.3]
    gd = GammaPDF()
    res = gd.fit(values)
    mu = gd.mean()
    assert np.abs(gd.mean() - np.mean(values)) < 1e-6
    print("Passed fitting problematic observations test!")

    # Test fitting multiple observations
    for n in range(2, 11):
        values = np.random.randint(1, 100, n) * 0.1
        gd = GammaPDF()
        res = gd.fit(values)
        assert np.abs(gd.mean() - np.mean(values)) < 1e-6
    print("Passed fitting multiple observations tests!")
