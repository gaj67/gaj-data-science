"""
This module implements the Beta distribution.

Note that we choose the logit function for the link function, which means that the link
parameter is not one of the natural parameters.
"""
import numpy as np
from scipy.special import digamma, polygamma

from core_dist import Distribution


class BetaDistribution(Distribution):

    def __init__(self, alpha, beta):
        """
        Initialises the Beta distribution.
        
        Inputs:
            - alpha (float): The first shape parameter.
            - beta (float): The second shape parameter.
        """
        super().__init__(alpha, beta)

    def mean(self):
        alpha, beta = self.parameters()
        return alpha / (alpha + beta)

    def variance(self):
        alpha, beta = self.parameters()
        nu = alpha + beta
        return alpha * beta / (nu**2 * (nu + 1))

    def natural_parameters(self):
        return self.parameters()

    def natural_variates(self, X):
        return np.array([np.log(X), np.log(1 - X)])

    def natural_means(self):
        alpha, beta = self.parameters()
        d_alpha = digamma(alpha)
        d_beta = digamma(beta)
        d_nu = digamma(alpha + beta)
        return np.array([d_alpha - d_nu, d_beta - d_nu])

    def natural_variances(self):
        alpha, beta = self.parameters()
        t_alpha = polygamma(1, alpha)
        t_beta = polygamma(1, beta)
        nt_nu =  -polygamma(1, alpha + beta)
        return np.array([
            [t_alpha + nt_nu, nt_nu],
            [nt_nu, t_beta + nt_nu]
        ])

    def link_parameter(self):
        alpha, beta = self.parameters()
        return np.log(alpha / beta)

    def link_variate(self, X):
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = self.natural_variates(X)
        return alpha * Y_alpha - beta * Y_beta

    def link_mean(self):
        alpha, beta = self.parameters()
        mu_alpha, mu_beta = self.natural_means()
        return alpha * mu_alpha - beta * mu_beta

    def link_variance(self):
        alpha, beta = self.parameters()
        params = np.array([alpha, -beta])
        Sigma = self.natural_variances()
        return params @ Sigma @ params
