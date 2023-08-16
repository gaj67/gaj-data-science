"""
This module implements the Beta distribution.

Note that we choose the logit function for the link function, which means that the link
parameter is not one of the natural parameters.
"""
import numpy as np
from numpy import ndarray
from scipy.stats import beta as beta_dist
from scipy.special import digamma, polygamma
from core_dist import ScalarPDF, RegressionPDF, Value, Collection
from stats_tools import logistic


class BetaDistribution(ScalarPDF):

    def __init__(self, alpha: Value, beta: Value):
        """
        Initialises the Beta distribution(s).

        Inputs:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

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
        return np.mean(beta_dist.logpdf(X, alpha, beta))

    def natural_parameters(self) -> Collection:
        return self.parameters()

    def natural_variates(self, X: Value) -> Collection:
        return (np.log(X), np.log(1 - X))

    def natural_means(self) -> Collection:
        alpha, beta = self.parameters()
        d_alpha = digamma(alpha)
        d_beta = digamma(beta)
        d_nu = digamma(alpha + beta)
        return (d_alpha - d_nu, d_beta - d_nu)

    def natural_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        t_alpha = polygamma(1, alpha)
        t_beta = polygamma(1, beta)
        nt_nu =  -polygamma(1, alpha + beta)
        return np.array([
            [t_alpha + nt_nu, nt_nu],
            [nt_nu, t_beta + nt_nu]
        ])

    def link_parameter(self) -> Value:
        alpha, beta = self.parameters()
        return np.log(alpha / beta)

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
            alpha**2 * Sigma[0, 0] + beta**2 * Sigma[1, 1]
            -2 * alpha * beta * Sigma[0, 1]
        )


class BetaRegression(RegressionPDF):

    def __init__(self, alpha: float, phi: ndarray):
        """
        Initialises the Beta distribution.

        Note that of the distributional parameters, alpha and beta, we have
        chosen alpha to be independent, and beta to be dependent on alpha and
        on the regression model.

        Input:
            - alpha (float): The independent distributional parameter.
            - phi (ndarray): The regression model parameters.
        """
        super().__init__(phi, alpha)

    def _distribution(self, eta: Value) -> ScalarPDF:
        alpha = self.independent_parameters()[0]
        beta = alpha * np.exp(-eta)
        return BetaDistribution(alpha, beta)

    def _independent_delta(self, X: Value, pdf: ScalarPDF) -> ndarray:
        # Independent parameter is alpha, with variate Y_alpha
        Y = pdf.natural_variates(X)[0]  # Y_alpha
        mu = pdf.natural_means()[0]  # E[Y_alpha]
        sigma_sq = pdf.natural_variances()[0, 0]  # Var[Y_alpha]
        return np.mean(Y - mu) / np.mean(sigma_sq)
