"""
This module defines the base class for a probability distribution function (PDF)
of a (discrete or continuous) scalar variate, in terms of one or more
distributional parameters.

Each parameter may independently be specified as either a scalar value or
a vector of values (see the Value type). This also holds true for
the value(s) of the response variate.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, Union, Tuple
from numpy import ndarray

import numpy as np


###############################################################################
# Useful types:

# A Vector is a 1D array
Vector: TypeAlias = ndarray

# A scalar is a float-like single value.
Scalar: TypeAlias = float  # 'float' really means 'float-like'

# A Value represents the 'value' of a single parameter or variate, which may in
# fact be either single-valued (i.e. scalar) or multi-valued (i.e. a vector of
# values).
Value: TypeAlias = Union[Scalar, Vector]

# A Values instance represents the 'value(s)' of one or more parameters or variates
# (each being either single-valued or multi-valued) in some fixed order.
Values: TypeAlias = Tuple[Value]

# A Values2d instance represents the 'value(s)' of one or more parameters or variates
# (each being either single-valued or multi-valued) in some fixed, two-dimensional order.
Values2d: TypeAlias = Tuple[Values]


###############################################################################
# Useful functions:


def to_value(value: object) -> Value:
    """
    Converts a scalar value or array-like of values to the Value type.
    """
    if not isinstance(value, ndarray):
        if hasattr(value, "__len__"):
            value = np.asarray(value)
        elif hasattr(value, "__iter__"):
            value = np.fromiter(value, float)
    if isinstance(value, ndarray) and len(value.shape) > 1:
        raise ValueError("Value must be scalar or vector-like")
    return value


def is_scalar(value: Value) -> bool:
    """
    Determines whether or not the argument is scalar-valued.

    Input:
        - value (float or ndarray): The value(s).
    Returns:
        - flag (bool): A value of True if the input is scalar,
            otherwise a value of False.
    """
    return not isinstance(value, ndarray) or len(value.shape) == 0


def is_vector(value: Value) -> bool:
    """
    Determines whether or not the argument is vector-valued.

    Input:
        - value (float or ndarray): The value(s).
    Returns:
        - flag (bool): A value of True if the input is a vector,
            otherwise a value of False.
    """
    return isinstance(value, ndarray) and len(value.shape) == 1


def is_divergent(value: Value) -> bool:
    """
    Determines whether the value(s) indicate divergence.

    Input:
        - value (float or ndarray): The value(s).
    Returns:
        - flag (bool): A value of True if there is any divergence,
            else False.
    """
    return np.any(np.isnan(value)) or np.any(np.isinf(value))


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

        _params = []
        size = 1
        for i, param in enumerate(params):
            if not is_scalar(param):
                if len(param.shape) != 1:
                    raise ValueError(f"Expected parameter {i} to be uni-dimensional")
                _len = len(param)
                if _len <= 0:
                    raise ValueError(f"Expected parameter {i} to be non-empty")
                if _len == 1:
                    # Take scalar value
                    param = param[0]
                elif size == 1:
                    size = _len
                elif size != _len:
                    raise ValueError(f"Expected parameter {i} to have length {size}")
            _params.append(param)
        self._params = tuple(_params)
        self._size = size

    def is_valid_parameters(self, *params: Values) -> bool:
        """
        Determines whether or not the given parameter values are viable.
        Specifically, there should be the correct number of parameters, and
        each parameter should have finite value(s) within appropriate bounds.

        Input:
            - params (tuple of float): The proposed parameter values.

        Returns:
            - flag (bool): A  value of True if the values are valid,
                else False.
        """
        # By default, just check for  divergence
        for value in params:
            if is_divergent(value):
                return False
        return True


###############################################################################
# Base distribution class:


class Distribution(Parameterised):
    """
    A parameterised probability distribution of a scalar variate, X.

    Each parameter may have either a single value or multiple values.
    If all parameters are single-valued, then only a single distribution
    is specified, and all computations, e.g. the distributional mean or
    variance, etc., will be single-valued.

    However, the use of one or more parameters with multiple values
    indicates a collection of distributions, rather than a single
    distribution. As such, all computations will be multi-valued
    rather than single-valued.
    """

    @abstractmethod
    def mean(self) -> Value:
        """
        Obtains the mean(s) of the distribution(s).

        Returns:
            - mu (float or ndarray): The mean value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Value:
        """
        Obtains the variance(s) of the distribution(s).

        Returns:
            - sigma_sq (float or ndarray): The variance(s).
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, data: Value) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - data (float or ndarray): The value(s) of the response variate.

        Returns:
            - log_prob (float or ndarray): The log-likelihood(s).
        """
        raise NotImplementedError
