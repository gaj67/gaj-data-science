"""
This module implements the Bernoulli distribution.

Note that we choose the link function to be the logit function, such that the natural
parameter and the link parameter are the same, and the natural variate and the link variate
are also the same.
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
        return np.array([[ self.variance() ]])

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
        Initialises the Bernoulli distribution.

        Input:
            - phi (ndarray): The regression parameters.
        """
        super().__init__(phi)

    def _distribution(self, eta: Value) -> ScalarPDF:
        mu = logistic(eta)
        return BernoulliDistribution(mu)

    def _independent_delta(self, X: Value, pdf: ScalarPDF) -> ndarray:
        return np.array([], dtype=float)
