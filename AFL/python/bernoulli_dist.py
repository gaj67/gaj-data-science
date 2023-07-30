"""
This module implements the Bernoulli distribution.

Note that we choose the link function to be the logit function, such that the natural
parameter and the link parameter are the same, and the natural variate and the link variate
are also the same.
"""
import numpy as np
from core_dist import Distribution


class BernoulliDistribution(Distribution):

    def __init__(self, theta):
        """
        Initialises the Bernoulli distribution.
        
        Input:
            - theta (float): The distributional parameter.
        """
        super().__init__(theta)

    def mean(self):
        return self.parameters()[0]

    def variance(self):
        mu = self.parameters()[0]
        return mu * (1 - mu)

    def natural_parameters(self):
        theta = self.parameters()[0]
        eta = np.log(theta / (1 - theta))
        return np.array([eta])

    def natural_variates(self, X):
        return np.array([X])

    def natural_means(self):
        return np.array([self.mean()])

    def natural_variances(self):
        return np.array([[self.variance()]])

    def link_parameter(self):
        return self.natural_parameters()[0]

    def link_variate(self, X):
        return X

    def link_mean(self):
        return self.mean()

    def link_variance(self):
        return self.variance()
