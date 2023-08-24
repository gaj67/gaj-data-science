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
"""
import numpy as np
from numpy import ndarray
from core_dist import ScalarPDF, RegressionPDF, Value, Collection
from stats_tools import logit


class BetaBernoulliDistribution(ScalarPDF):
    def __init__(self, alpha: Value, beta: Value):
        """
        Initialises the Beta-Bernoulli distribution(s).

        Input:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

    def mean(self) -> Value:
        alpha, beta = self.parameters()
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        mu = self.mean()
        return mu * (1 - mu)

    def log_prob(self, X: Value) -> Value:
        mu = self.mean()
        eta = logit(mu)
        return np.log(1 - mu) + X * eta

    def natural_parameters(self) -> Collection:
        return (self.link_parameter(),)

    def natural_variates(self, X: Value) -> Collection:
        return (X,)

    def natural_means(self) -> Collection:
        return (self.mean(),)

    def natural_variances(self) -> ndarray:
        return np.array([[self.variance()]])

    def link_parameter(self) -> Value:
        return logit(self.mean())

    def link_variate(self, X: Value) -> Value:
        return X

    def link_mean(self) -> Value:
        return self.mean()

    def link_variance(self) -> Value:
        return self.variance()


class BetaBernoulliRegression(RegressionPDF):
    def __init__(self, phi: ndarray):
        """
        Initialises the Beta-Bernoulli regression distribution.

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
        return BetaBernoulliDistribution(alpha, beta)

    def _independent_delta(self, X: Value, pdf: ScalarPDF) -> ndarray:
        # Independent parameter is alpha, with variate Y_alpha = X / alpha
        alpha = self.independent_parameters()[0]
        mu = pdf.mean()  # E[Y_alpha] * alpha
        sigma_sq = mu * (1 - mu)  # Var[Y_alpha] * alpha^2
        return alpha * np.mean(X - mu) / np.mean(sigma_sq)
