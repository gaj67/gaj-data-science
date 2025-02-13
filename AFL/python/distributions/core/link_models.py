"""
This module provides a number of pre-built link models useful for regression.
"""

import numpy as np
from scipy.special import logit, expit as logistic

from .distribution import TransformDistribution
from .parameterised import guard_prob, guard_pos
from .data_types import Values, Values2d


########################################################
# Logit link models


class LogitLink11(TransformDistribution):
    """
    Implements a one-parameter logit link model,
    namely:

           eta = logit(theta) ;
        => theta = logistic(eta) ,

    where the underlying parameter, theta, represents
    a probability or proportion.
    """

    def num_links(self) -> int:
        return 1

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


class LogitLink21(TransformDistribution):
    """
    Implements a two-parameter logit link model,
    namely:

           eta = logit(theta) , psi = alpha ;
        => theta = logistic(eta) , alpha = psi ,

    where the underlying parameter, theta, represents
    a probability or proportion, and the independent
    parameter, alpha, represents a rate or count.
    """

    def num_links(self) -> int:
        return 1

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
# Log link models


class LogLink11(TransformDistribution):
    """
    Implements a one-parameter log link model,
    namely:

           eta = log(lambda) ;
        => lambda = exp(eta) ,

    where the underlying parameter, lambda, represents
    a positive rate or scale.
    """

    def num_links(self) -> int:
        return 1

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


class LogLink22(TransformDistribution):
    """
    Implements a two-parameter log link model,
    namely:

           eta_1 = log(alpha) , eta_2 = log(beta) ;
        => alpha = exp(eta_1) , beta = exp(eta_2) ,

    where the underlying parameters, alpha and beta, 
    represent positive rates or scales.
    """

    def num_links(self) -> int:
        return 2

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 2 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha = guard_pos(std_params[0])
        beta = guard_pos(std_params[1])
        eta_1 = np.log(alpha)
        eta_2 = np.log(beta)
        return (eta_1, eta_2)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta_1, eta_2 = alt_params
        alpha = np.exp(eta_1)
        beta = np.exp(eta_2)
        return (alpha, beta)

    def compute_jacobian(self) -> Values2d:
        alpha, beta = self.underlying().get_parameters()
        return ((alpha, 0), (0, beta))


class LogLinkn2(TransformDistribution):
    """
    Implements a two-parameter log link model,
    namely:

           eta_1 = log(alpha) , eta_2 = log(beta) ;
        => alpha = exp(eta_1) , beta = exp(eta_2) ,

    where the underlying parameters, alpha and beta, 
    represent positive rates or scales.
    
    Any remaining parameters are simply passed through
    as independent parameters, psi.
    """

    def num_links(self) -> int:
        return 2

    def check_parameters(self, *params: Values) -> bool:
        return len(params) >= 2 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha = guard_pos(std_params[0])
        beta = guard_pos(std_params[1])
        psi = std_params[2:]
        eta_1 = np.log(alpha)
        eta_2 = np.log(beta)
        return (eta_1, eta_2, *psi)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta_1, eta_2, *psi = alt_params
        alpha = np.exp(eta_1)
        beta = np.exp(eta_2)
        return (alpha, beta, *psi)

    def compute_jacobian(self) -> Values2d:
        alpha, beta, *psi = self.underlying().get_parameters()
        jac = np.eye(len(psi) + 2)
        jac[0, 0] = alpha
        jac[1, 1] = beta
        return tuple(tuple(row) for row in jac)


########################################################
# Log-ratio link models (two parameters):


class LogRatioLink21a(TransformDistribution):
    """
    Implements a two-parameter log link function:

           eta = log(alpha/beta) , psi = alpha ;
        => alpha = psi , beta = psi*exp(-eta) ,

    where the underlying parameters, alpha and beta,
    represent positive shapes or rates.
    """

    def num_links(self) -> int:
        return 1

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        eta, psi = params
        return np.all(psi > 0)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha = guard_pos(std_params[0])
        beta = guard_pos(std_params[1])
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


class LogRatioLink21b(TransformDistribution):
    """
    Implements a two-parameter log link function:

           eta = log(alpha/beta) , psi = beta ;
        => alpha = psi*exp(eta) , beta = psi ,

    where the underlying parameters, alpha and beta,
    represent positive shapes or rates.
    """

    def num_links(self) -> int:
        return 1

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        eta, psi = params
        return np.all(psi > 0)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha = guard_pos(std_params[0])
        beta = guard_pos(std_params[1])
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


class LogRatioLink22(TransformDistribution):
    """
    Implements a two-parameter log link function:

           eta_1 = log(alpha/beta) , eta_2 = log(alpha*beta) ;
        => alpha = exp((eta_2+eta_1)/2) , beta = exp((eta_2-eta_1)/2) ,

    where the underlying parameters, alpha and beta,
    represent positive shapes or rates.
    """

    def num_links(self) -> int:
        return 2

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 2 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha = guard_pos(std_params[0])
        beta = guard_pos(std_params[1])
        eta_1 = np.log(alpha / beta)
        eta_2 = np.log(alpha * beta)
        return (eta_1, eta_2)

    def invert_transform(self, *alt_params: Values) -> Values:
        eta_1, eta_2 = eta_1
        alpha = np.exp(0.5 * (eta_2 + eta_1))
        beta = np.exp(0.5 * (eta_2 - eta_1))
        return (alpha, beta)

    def compute_jacobian(self) -> Values2d:
        alpha, beta = self.underlying().get_parameters()
        return ((0.5 * alpha, -0.5 * beta), (0.5 * alpha, 0.5 * beta))


########################################################
# Special link models


class SwapLink20(TransformDistribution):
    """
    Implements a special link model that simply swaps
    the order of parameters, namely:

           psi1 = beta ; psi2 = alpha
        => alpha = psi2; beta = psi1

    where the underlying parameters are alpha and beta.
    
    It is assumed that neither parameter is intrinsically
    a link parameter. To change this behaviour, use 
    core.distribution.add_link_model() to add another
    link model over this one.
    """

    def num_links(self) -> int:
        return 0

    def check_parameters(self, *params: Values) -> bool:
        return len(params) == 2 and super().check_parameters(*params)

    def apply_transform(self, *std_params: Values) -> Values:
        alpha, beta = std_params
        psi1 = beta
        psi2 = alpha
        return (psi1, psi2)

    def invert_transform(self, *alt_params: Values) -> Values:
        psi1, psi2 = alt_params
        alpha = psi2
        beta = psi1
        return (alpha, beta)

    def compute_jacobian(self) -> Values2d:
        return ((0, 1), (1, 0))


class ReverseLinkn0(TransformDistribution):
    """
    Implements a special link model that simply reverses
    the order of parameters.
    
    It is assumed that neither parameter is intrinsically
    a link parameter. To change this behaviour, use 
    core.distribution.add_link_model() to add another
    link model over this one.
    """

    def num_links(self) -> int:
        return 0

    def apply_transform(self, *std_params: Values) -> Values:
        return tuple(reversed(std_params))

    def invert_transform(self, *alt_params: Values) -> Values:
        return tuple(reversed(alt_params))

    def compute_jacobian(self) -> Values2d:
        std_params = self.underlying().get_parameters()
        jac = np.eye(len(std_params))[:, ::-1]
        return tuple(tuple(row) for row in jac)
