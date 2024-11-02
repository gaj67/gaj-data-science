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

    @staticmethod
    @abstractmethod
    def default_parameters() -> Values:
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

    @abstractmethod
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
        raise NotImplementedError


###############################################################################
# Base implementation class:


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
        """
        Provides the values of the distributional parameters.

        Returns:
            - params (tuple of float or ndarray): The parameter values.
        """
        return self._params

    def set_parameters(self, *params: Values):
        """
        Overrides the distributional parameter value(s).

        Invalid parameters will cause an exception.

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")
        self._params = params

    def is_valid_parameters(self, *params: Values) -> bool:
        # By default, just check for dimensionality and divergence
        for value in params:
            if not is_scalar(value) and not is_vector(value):
                return False
            if is_divergent(value):
                return False
        return True


###############################################################################
# Regression implementation class:


# Indicates that the regression weights have not yet been specified
UNSPECIFIED_REGRESSION = np.array([])


class RegressionParameters(Parameters):
    """
    Implements a container for regression parameters and optionally
    other independent parameters.

    Specifically, the regression parameters take the form of a vector,
    which is always placed first amongst all parameters.

    Note: Use the UNSPECIFIED_REGRESSION constant to indicate that the
    regression parameters are unknown, and need to be set or estimated.
    """

    def regression_parameters(self) -> Vector:
        """
        Obtains the regression parameters.

        Returns:
            - reg_params (vector): The value(s) of the regression parameter(s).
        """
        return self.parameters()[0]

    def independent_parameters(self) -> Values:
        """
        Obtains the independent parameters.

        Returns:
            - indep_params (tuple of float or vector): The value(s) of the independent
                parameter(s), if any.
        """
        return self.parameters()[1:]

    def is_valid_parameters(self, *params: Values) -> bool:
        # Must have a vector first parameter!
        if len(params) == 0:
            return False
        _iter = iter(params)
        phi = next(_iter)
        if not is_vector(phi) or is_divergent(phi):
            return False
        # Any remaining parameters must be scalar or vector
        for value in _iter:
            if not is_scalar(value) and not is_vector(value):
                return False
            if is_divergent(value):
                return False
        return True
