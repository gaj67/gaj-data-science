"""
This module provides a number of pre-built link models useful for regression.
"""

import numpy as np
from scipy.special import logit, expit as logistic

from .distribution import TransformDistribution
from .parameterised import guard_prob, guard_pos
from .data_types import Values, Values2d


########################################################
# Logit link model (one parameter):


class LogitLink1(TransformDistribution):
    """
    Implements a one-parameter logit link model,
    namely:

           eta = logit(theta) ;
        => theta = logistic(eta) ,

    where the underlying parameter, theta, represents
    a probability or proportion.
    """

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 1 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        theta = guard_prob(std_params[0])
        eta = logit(theta)
        return (eta,)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta = alt_params[0]
        theta = logistic(eta)
        return (theta,)

    def compute_jacobian(self) -> Values2d:
        theta = self.underlying().get_parameters()[0]
        d_theta_d_eta = theta * (1 - theta)
        return ((d_theta_d_eta,),)


########################################################
# Logit link model (two parameters):


class LogitLink2(TransformDistribution):
    """
    Implements a two-parameter logit link model,
    namely:

           eta = logit(theta) , psi = alpha ;
        => theta = logistic(eta) , alpha = psi ,

    where the underlying parameter, theta, represents
    a probability or proportion, and the independent
    parameter, alpha , represents a rate or count.
    """

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 2 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        theta, alpha = std_params
        eta = logit(guard_prob(theta))
        psi = alpha
        return (eta, psi)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta, psi = alt_params
        theta = logistic(eta)
        alpha = psi
        return (theta, alpha)

    def compute_jacobian(self) -> Values2d:
        theta, alpha = self.underlying().get_parameters()
        return ((theta * (1 - theta), 0), (0, 1))



########################################################
# Log link model (one parameter):


class LogLink1(TransformDistribution):
    """
    Implements a one-parameter log link model,
    namely:

           eta = log(lambda) ;
        => lambda = exp(eta) ,

    where the underlying parameter, lambda, represents
    a positive rate or scale.
    """

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 1 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        _lambda = guard_pos(std_params[0])
        eta = np.log(_lambda)
        return (eta,)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta = alt_params[0]
        _lambda = np.exp(eta)
        return (_lambda,)

    def compute_jacobian(self) -> Values2d:
        _lambda = self.underlying().get_parameters()[0]
        return ((_lambda,),)


########################################################
# Log-ratio link models (two parameters):


class LogRatioLink2a(TransformDistribution):
    """
    Implements a two-parameter log link function:

           eta = log(alpha/beta) , psi = alpha ;
        => alpha = psi , beta = psi*exp(-eta) ,

    where the underlying parameters, alpha and beta,
    represent positive shapes or rates.
    """

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        eta, psi = params
        return np.all(psi > 0)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha, beta = std_params
        eta = np.log(alpha / beta)
        psi = alpha
        return (eta, psi)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta, psi = alt_params
        alpha = psi
        beta = psi * np.exp(-eta)
        return (alpha, beta)

    def compute_jacobian(self) -> Values2d:
        alpha, beta = self.underlying().get_parameters()
        return ((0, -beta), (1, beta / alpha))


class LogRatioLink2b(TransformDistribution):
    """
    Implements a two-parameter log link function:

           eta = log(alpha/beta) , psi = beta ;
        => alpha = psi*exp(eta) , beta = psi ,

    where the underlying parameters, alpha and beta,
    represent positive shapes or rates.
    """

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        eta, psi = params
        return np.all(psi > 0)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha, beta = std_params
        eta = np.log(alpha / beta)
        psi = beta
        return (eta, psi)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta, psi = alt_params
        alpha = psi * np.exp(eta)
        beta = psi
        return (alpha, beta)

    def compute_jacobian(self) -> Values2d:
        alpha, beta = self.underlying().get_parameters()
        return ((alpha, 0), (alpha / beta, 1))


class LogRatioLink2ab(TransformDistribution):
    """
    Implements a two-parameter log link function:

           eta = log(alpha/beta) , psi = log(alpha*beta) ;
        => alpha = exp((psi+eta)/2) , beta = exp((psi-eta)/2) ,

    where the underlying parameters, alpha and beta,
    represent positive shapes or rates.
    """

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 2 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha, beta = std_params
        eta = np.log(alpha / beta)
        psi = np.log(alpha * beta)
        return (eta, psi)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta, psi = alt_params
        alpha = np.exp(0.5 * (psi + eta))
        beta = np.exp(0.5 * (psi - eta))
        return (alpha, beta)

    def compute_jacobian(self) -> Values2d:
        alpha, beta = self.underlying().get_parameters()
        return ((0.5 * alpha, -0.5 * beta), (0.5 * alpha, 0.5 * beta))
