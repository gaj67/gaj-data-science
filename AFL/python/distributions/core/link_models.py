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

        - eta = logit(theta),
        - theta = logistic(eta),

    where theta is a probability or proportion.
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
# Log link model (one parameter):


class LogLink1(TransformDistribution):
    """
    Implements a one-parameter log link model,
    namely:

        - eta = log(lambda),
        - laambda = exp(eta),

    where lambda is a positive rate or scale.
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
