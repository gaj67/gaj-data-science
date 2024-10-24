"""
This module defines the base class for an object that has one or more
parameters.

Each parameter may independently be specified as either a scalar value or
a vector of values (see the data_types.Value type).
"""

from abc import ABC, abstractmethod

from .data_types import (
    Values,
    is_scalar,
    is_vector,
    is_divergent,
)

###############################################################################
# Base parameter class:


class Parameterised(ABC):
    """
    Implements a mutable holder of parameters.
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

    def __init__(self, *params: Values):
        """
        Initialises the instance with the given parameter value(s).
        Each parameter may have either a single value or multiple values.

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
