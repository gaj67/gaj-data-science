"""
This module defines the base types needed for numerical analysis on
a probability distribution function (PDF) of a discrete or continuous,
scalar variate, in terms of one or more distributional parameters.

Each parameter or variate may independently be specified as either
a scalar value or a vector of values (see the Value type). 
Covariates may also be a matrix.
"""

from typing import TypeAlias, Union, Tuple
from numpy.typing import NDArray, ArrayLike
from numpy import ndarray

import numpy as np


###############################################################################
# Useful types:

# A float-like single value.
Scalar: TypeAlias = Union[float, int]

# A 1D array of scalar value(s).
Vector: TypeAlias = NDArray[Scalar]

# A 2D array of scalar values
Matrix: TypeAlias = NDArray[NDArray[Scalar]]

# The 'value' of a single parameter or variate, which may in fact be either a
# scalar value or a vector of value(s).
Value: TypeAlias = Union[Scalar, Vector]

# An extension to permit array-like values as vectors.
ValueLike: TypeAlias = Union[Value, ArrayLike]

# The value(s) of zero, one or more parameters or variates.
Values: TypeAlias = Tuple[Value]

# The value(s) of zero, one or more parameters or variates in some fixed,
# two-dimensional order.
Values2d: TypeAlias = Tuple[Values]


###############################################################################
# Useful value-checking functions:

# NOTE: We still have an issue with strict type checking, e.g. an np.float32 is
# not an instance of float, nor is an np.int64 an instance of int.
# Consequently, we simply check for ndarray or sometimes array-like.


def is_scalar(value: Value) -> bool:
    """
    Determines whether or not the argument is scalar-valued.

    Input:
        - value (float-like or vector): The value(s).
    Returns:
        - flag (bool): A value of True if the input is scalar,
            otherwise a value of False.
    """
    return not isinstance(value, ndarray) or len(value.shape) == 0


def is_vector(value: Value) -> bool:
    """
    Determines whether or not the argument is vector-valued.

    Input:
        - value (float-like or vector): The value(s).
    Returns:
        - flag (bool): A value of True if the input is a vector,
            otherwise a value of False.
    """
    return isinstance(value, ndarray) and len(value.shape) == 1


def is_divergent(value: Value) -> bool:
    """
    Determines whether the value(s) indicate divergence.

    Input:
        - value (float-like or vector): The value(s).
    Returns:
        - flag (bool): A value of True if there is any divergence,
            else False.
    """
    return np.any(np.isnan(value)) or np.any(np.isinf(value))


def is_scalars(*values: Values) -> bool:
    """
    Determines whether or not the given values are all scalar.

    Input:
        - values (tuple of float-like or vector): The input values.
    Returns:
        - flag (bool): A value of True if all input values are scalar,
            otherwise False.
    """
    return np.all(list(map(is_scalar, values)))


###############################################################################
# Useful value-conversion functions:


def to_value(value: ValueLike) -> Value:
    """
    Converts a scalar or vector-like of values to the Value type.

    Input:
        - value (float-like or vector-like): The input value(s).

    Returns:
        - output (float-like or vector): The output value(s).
    """
    if isinstance(value, ndarray):
        if len(value.shape) > 1:
            raise ValueError("Value must be scalar or vector-like")
        return value
    if hasattr(value, "__len__"):
        output = np.asarray(value)
        if len(output.shape) != 1:
            raise ValueError("Value must be scalar or vector-like")
    if hasattr(value, "__iter__"):
        return np.fromiter(value, float)
    # Assume scalar
    return value


def to_vector(value: ValueLike) -> Vector:
    """
    Converts the value(s) to an ndarray vector.

    Input:
        - value (float-like or vector-like): The input value(s).

    Returns:
        - vec (ndarray): The output vector.
    """
    if isinstance(value, ndarray):
        if len(value.shape) == 0:
            vec = np.array([value])
        else:
            vec = value
    elif hasattr(value, "__len__"):
        vec = np.asarray(value)
    elif hasattr(value, "__iter__"):
        vec = np.fromiter(value, float)
    else:
        vec = np.array([value])
    if len(vec.shape) != 1:
        raise ValueError("Value must be scalar or vector-like")
    return vec


def to_matrix(values: Values, n_dim: int) -> Matrix:
    """
    Converts the input tuple into an ndarray matrix.

    Input:
        - values (tuple of float-like or vector): The input values.
        - n_dim (int): The required row dimension.

    Returns:
        - mat (ndarray): The output matrix.
    """

    def convert(value: Value) -> ndarray:
        if is_scalar(value):
            return np.array([value] * n_dim)
        if len(value.shape) != 1 or value.shape[0] != n_dim:
            raise ValueError("Incompatible dimensions!")
        return value

    return np.column_stack([convert(v) for v in values])
