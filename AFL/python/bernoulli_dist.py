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
from core_dist import ScalarPDF, RegressionPDF, Value, Collection
from stats_tools import logistic, logit


class BernoulliDistribution(ScalarPDF):
    def __init__(self, theta: Value):
        """
        Initialises the Bernoulli distribution(s).

        Input:
            - theta (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta)

    def mean(self) -> Value:
        # Mean is just the parameter theta
        return self.parameters()[0]

    def variance(self) -> Value:
        mu = self.mean()
        return mu * (1 - mu)

    def log_prob(self, X: Value) -> Value:
        theta = self.parameters()[0]
        eta = logit(theta)
        return np.log(1 - theta) + X * eta

    def natural_parameters(self) -> Collection:
        return (self.link_parameter(),)

    def natural_variates(self, X: Value) -> Collection:
        return (X,)

    def natural_means(self) -> Collection:
        return (self.mean(),)

    def natural_variances(self) -> ndarray:
        return np.array([[self.variance()]])

    def link_parameter(self) -> Value:
        theta = self.parameters()[0]
        eta = logit(theta)
        return eta

    def link_variate(self, X: Value) -> Value:
        return X

    def link_mean(self) -> Value:
        return self.mean()

    def link_variance(self) -> Value:
        return self.variance()


class BernoulliRegression(RegressionPDF):
    def __init__(self, phi: ndarray):
        """
        Initialises the Bernoulli regression distribution.

        Input:
            - phi (ndarray): The regression parameters.
        """
        super().__init__(phi)

    def _distribution(self, eta: Value) -> ScalarPDF:
        mu = logistic(eta)
        return BernoulliDistribution(mu)
