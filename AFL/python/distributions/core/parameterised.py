"""
This module defines the base class for an object that has one or more
parameters.

Each parameter may independently be specified as either a scalar value or
a vector of values (see the data_types.Value type).
"""

from abc import ABC, abstractmethod

import numpy as np

from .data_types import (
    Value,
    Values,
    Vector,
    is_scalar,
    is_vector,
    is_divergent,
)

###############################################################################
# Handy methods for guarding against impossible parameter values:


def guard_prob(value: Value) -> Value:
    """
    Guard against extreme probability values.

    Input:
        - value (float or vector): The value(s) to be checked.
    Returns:
        - value' (float or vector): The adjusted value(s).
    """
    if is_scalar(value):
        return 1e-30 if value <= 0.0 else 1 - 1e-10 if value >= 1.0 else value
    value = value.copy()
    value[value <= 0] = 1e-30
    value[value >= 1] = 1 - 1e-10
    return value


def guard_pos(value: Value) -> Value:
    """
    Guard against values going non-positive.

    Input:
        - value (float or vector): The value(s) to be checked.
    Returns:
        - value' (float or vector): The adjusted value(s).
    """
    if is_scalar(value):
        return 1e-30 if value <= 0.0 else value
    value = value.copy()
    value[value <= 0] = 1e-30
    return value


###############################################################################
# Base parameter class:


class Parameterised(ABC):
    """
    Interface for a mutable holder of parameters.
    """

    @abstractmethod
    def default_parameters(self) -> Values:
        """
        Provides default (scalar) values of the distributional parameters.

        Returns:
            - params (tuple of float): The default parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> Values:
        """
        Provides the values of the distributional parameters.

        Returns:
            - params (tuple of float or ndarray): The parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, *params: Values):
        """
        Overrides the distributional parameter value(s).

        Invalid parameters will cause an exception.

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        raise NotImplementedError

    def is_valid_parameters(self, *params: Values) -> bool:
        """
        Determines whether or not the given parameter values are viable.
        Specifically, there should be the correct number of parameters, and
        each parameter should have finite value(s) within appropriate bounds.

        Input:
            - params (tuple of float or vector): The proposed parameter values.

        Returns:
            - flag (bool): A  value of True if the values are all valid,
                else False.
        """
        # By default, just check for dimensionality and divergence
        for value in params:
            if not is_scalar(value) and not is_vector(value):
                return False
            if is_divergent(value):
                return False
        return True


###############################################################################
# Standard implementation class:


class Parameters(Parameterised):
    """
    Implements a holder of parameters with mutable values.

    Each parameter may have either a single (scalar) value
    or multiple (vector) values.
    """

    def __init__(self, *params: Values):
        """
        Initialises the instance with the given parameter value(s).

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        if len(params) == 0:
            self.set_parameters(*self.default_parameters())
        else:
            self.set_parameters(*params)

    def parameters(self) -> Values:
        return self._params

    def set_parameters(self, *params: Values):
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")
        self._params = params


###############################################################################
# Special implementation class for regression:


# Indicates that the regression weights have not yet been specified
UNSPECIFIED_REGRESSION = np.array([])

# The default value of the link parameter
DEFAULT_LINK = 0.0


class RegressionParameters(Parameterised):
    """
    An implementation for accessing regression parameters,
    where the additional, independent parameters are accessed
    through an underlying link model.
    """

    def __init__(
        self, link: Parameterised, reg_params: Vector = UNSPECIFIED_REGRESSION
    ):
        """
        Initialises the regression with the underlying link model,
        and optionally the regression parameters.

        Input:
            - link (parameterised): The link model.
            - reg_params (vector, optional): The regression parameters.
        """
        self._link = link
        self.set_parameters(reg_params, *link.parameters()[1:])

    def underlying(self) -> Parameterised:
        """
        Obtains the underlying regression link model.

        Returns:
            - link (linkable): The link model.
        """
        return self._link

    def regression_parameters(self) -> Vector:
        """
        Obtains the regression parameters.

        Returns:
            - reg_params (vector): The value(s) of the regression parameter(s).
        """
        return self._phi

    def independent_parameters(self) -> Values:
        """
        Obtains the independent parameters.

        Returns:
            - indep_params (tuple of float or vector): The value(s) of the independent
                parameter(s), if any.
        """
        return self.underlying().parameters()[1:]

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (UNSPECIFIED_REGRESSION,) + self.underlying().default_parameters()[1:]

    def parameters(self) -> Values:
        return (self._phi,) + self.independent_parameters()

    def set_parameters(self, *params: Values):
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")
        self._phi = params[0]
        self.underlying().set_parameters(DEFAULT_LINK, *params[1:])

    def is_valid_parameters(self, *params: Values) -> bool:
        # Must have a vector first parameter!
        if len(params) == 0 or not is_vector(params[0]):
            return False
        if not super().is_valid_parameters(params[0]):
            return False
        return self.underlying().is_valid_parameters(DEFAULT_LINK, *params[1:])
