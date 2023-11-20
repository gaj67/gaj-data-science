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
from numpy import ndarray
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, polygamma
from core_dist import ScalarPDF, RegressionPDF, Value, Collection


class GammaDistribution(ScalarPDF):
    def __init__(self, alpha: Value, beta: Value):
        """
        Initialises the Gamma distribution(s).

        Inputs:
            - alpha (float or ndarray): The shape parameter value(s).
            - beta (float or ndarray): The rate parameter value(s).
        """
        super().__init__(alpha, beta)

    def mean(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta

    def variance(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / beta**2

    def log_prob(self, X: Value) -> Value:
        alpha, beta = self.parameters()
        return gamma_dist.logpdf(X, alpha, scale=1.0 / beta)

    def natural_parameters(self) -> Collection:
        return self.parameters()

    def natural_variates(self, X: Value) -> Collection:
        return (np.log(X), -X)

    def natural_means(self) -> Collection:
        alpha, beta = self.parameters()
        return (digamma(alpha) - np.log(beta), -alpha / beta)

    def natural_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        cov = -1 / beta
        return np.array([[polygamma(1, alpha), cov], [cov, alpha / beta**2]])

    def link_parameter(self) -> Value:
        return np.log(self.mean())

    def link_variate(self, X: Value) -> Value:
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = self.natural_variates(X)
        return alpha * Y_alpha - beta * Y_beta

    def link_mean(self) -> Value:
        alpha, beta = self.parameters()
        mu_alpha, mu_beta = self.natural_means()
        return alpha * mu_alpha - beta * mu_beta

    def link_variance(self) -> Value:
        alpha, beta = self.parameters()
        Sigma = self.natural_variances()
        return (
            alpha**2 * Sigma[0, 0]
            + beta**2 * Sigma[1, 1]
            - 2 * alpha * beta * Sigma[0, 1]
        )


class GammaRegression(RegressionPDF):
    def __init__(self, alpha: float, phi: ndarray):
        """
        Initialises the Gamma regression distribution.

        Note that of the distributional parameters, alpha and beta, we have
        chosen beta to be independent, and alpha to be dependent on beta and
        on the regression model.

        Input:
            - beta (float): The independent rate parameter.
            - phi (ndarray): The regression model parameters.
        """
        super().__init__(phi, beta)

    def _distribution(self, eta: Value) -> ScalarPDF:
        beta = self.independent_parameters()[0]
        alpha = beta * np.exp(eta)
        return GammaDistribution(alpha, beta)

    def _independent_delta(self, X: Value, W: Value, pdf: ScalarPDF) -> ndarray:
        # Independent parameter is beta, with variate Y_beta = -X
        beta = self.independent_parameters()[0]
        mu = pdf.mean()  # -E[Y_beta]
        return beta * np.sum(W * (mu - X)) / np.sum(W * mu)
