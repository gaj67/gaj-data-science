"""
This module defines the base types needed for numerical analysis on
a probability distribution function (PDF) of a discrete or continuous,
scalar variate, in terms of one or more distributional parameters.

Each parameter or variate may independently be specified as either
a scalar value or a vector of values (see the Value type). 
Covariates may also be a matrix.
"""

from typing import TypeAlias, Union, Tuple, Any
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

# The value(s) of zero, one or more parameters or variates.
Values: TypeAlias = Tuple[Value]

# The value(s) of zero, one or more parameters or variates in some fixed,
# two-dimensional order.
Values2d: TypeAlias = Tuple[Values]

# An extension to permit array-like values as vectors.
VectorLike: TypeAlias = Union[Scalar, Vector, ArrayLike]

# An extension to permit array-like values as matrices.
MatrixLike: TypeAlias = Union[Scalar, Vector, Matrix, ArrayLike]


###############################################################################
# Useful value-checking functions:

# NOTE: We still have an issue with strict type checking, e.g. an np.float32 is
# not an instance of float, nor is an np.int64 an instance of int.
# Consequently, we simply check for ndarray or array-like.


def is_scalar(value: Any) -> bool:
    """
    Determines whether or not the argument is scalar-valued.

    Input:
        - value (any): The value(s).
    Returns:
        - flag (bool): A value of True if the input is scalar,
            otherwise a value of False.
    """
    if isinstance(value, ndarray):
        return len(value.shape) == 0
    return not hasattr(value, "__len__") and not hasattr(value, "__iter__")


def is_vector(value: Any) -> bool:
    """
    Determines whether or not the argument is vector-valued.

    Input:
        - value (any): The value(s).
    Returns:
        - flag (bool): A value of True if the input is a vector,
            otherwise a value of False.
    """
    return isinstance(value, ndarray) and len(value.shape) == 1


def is_matrix(value: Any) -> bool:
    """
    Determines whether or not the argument is matrix-valued.

    Input:
        - value (any): The value(s).
    Returns:
        - flag (bool): A value of True if the input is a matrix,
            otherwise a value of False.
    """
    return isinstance(value, ndarray) and len(value.shape) == 2


def is_divergent(value: Any) -> bool:
    """
    Determines whether the value(s) indicate divergence.

    Input:
        - value (any): The value(s).
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


def to_vector(value: VectorLike, n_dim: int = -1) -> Vector:
    """
    Converts the value(s) into an ndarray vector.

    Input:
        - value (vector-like): The input value(s).
        - n_rows (int, optional): The required number of elements.

    Returns:
        - vec (ndarray): The output vector.
    """
    vec = np.asarray(value)
    if len(vec.shape) == 0:
        vec = np.array([value])
    if len(vec.shape) != 1:
        raise ValueError("Value must vector-like!")
    if n_dim >= 0 and len(vec) != n_dim:
        raise ValueError("Incompatible dimensions!")
    return vec


def to_matrix(value: MatrixLike, n_rows: int = -1, n_cols: int = -1) -> Matrix:
    """
    Converts the value(s) into an ndarray matrix.

    Input:
        - value (matrix-like): The input value(s).
        - n_rows (int, optional): The required number of rows.
        - n_cols (int, optional): The required number of columns.

    Returns:
        - mat (ndarray): The output matrix.
    """
    mat = np.asarray(value)
    if len(mat.shape) == 0:
        mat = np.array([[mat]])
    elif len(mat.shape) == 1:
        if n_cols >= 0:
            if len(mat) == n_cols:
                mat = mat.reshape((1, -1))
            else:
                mat = mat.reshape((-1, 1))
        elif n_rows >= 0:
            if len(mat) == n_rows:
                mat = mat.reshape((-1, 1))
            else:
                mat = mat.reshape((1, -1))
        else:
            raise ValueError("Ambiguous dimensions!")
    if len(mat.shape) != 2:
        raise ValueError("Value must matrix-like!")
    if n_rows >= 0 and mat.shape[0] != n_rows:
        raise ValueError("Incompatible row dimensions!")
    if n_cols >= 0 and mat.shape[1] != n_cols:
        raise ValueError("Incompatible column dimensions!")
    return mat


def values_to_matrix(values: Values, n_rows: int) -> Matrix:
    """
    Converts the input tuple into an ndarray matrix.

    Input:
        - values (tuple of float-like or vector): The input values.
        - n_rows (int): The required row dimension.

    Returns:
        - mat (ndarray): The output matrix.
    """

    def convert(value: Value) -> ndarray:
        if is_scalar(value):
            return np.array([value] * n_rows)
        if len(value.shape) != 1 or value.shape[0] != n_rows:
            raise ValueError("Incompatible dimensions!")
        return value

    return np.column_stack([convert(v) for v in values])


def to_value(value: VectorLike) -> Value:
    """
    Converts the value(s) into a scalar or vector.

    Input:
        - value (float-like or vector-like): The input value(s).

    Returns:
        - output (float-like or vector-like): The output value(s)
    """
    return as_value(to_vector(value))


def as_value(value: Vector) -> Value:
    """
    If possible, converts a singleton vector into a scalar.
    Otherwise, the vector is retained.

    Input:
        - value (vector): The vector value(s).

    Returns:
        - value' (float or vector): The scalar or vector value.
    """
    return value[0] if len(value) == 1 else value


def mean_value(weights: Vector, value: Value) -> float:
    """
    Computes the weighted mean of the scalar or vector value.

    Input:
        - weights (ndarray): The vector of weights.
        - values (float-like or vector): The scalar or vector value.

    Returns:
        - mean (float): The value mean.
    """
    return (weights @ to_vector(value)) / np.sum(weights)


def mean_values(weights: Vector, values: Values) -> Vector:
    """
    Computes the weighted mean of the scalar or vector values.

    Input:
        - weights (ndarray): The vector of weights.
        - values (tuple of float-like or vector): The scalar or vector values.

    Returns:
        - means (ndarray): The vector of value means.
    """
    return (weights @ values_to_matrix(values, len(weights))) / np.sum(weights)


def mult_rmat_vec(mat: Values2d, vec: Values) -> Values:
    """
    Multiplies each row of a matrix-like by a vector-like,
    i.e. computes X @ y.

    Input:
        - mat (matrix-like of scalar or vector): The matrix.
        - vec (tuple of scalar or vector): The vector.

    Returns:
        - vec' (tuple of float or vector): The vector result.
    """
    return tuple(sum(x * y for x, y in zip(row, vec)) for row in mat)


def mult_rmat_rmat(mat1: Values2d, mat2: Values2d) -> Values2d:
    """
    Multiplies the rows of the first matrix-like by the rows
    of the second matrix-like, i.e. computes X @ Y.T.

    Input:
        - mat1 (matrix-like of scalar or vector): The left matrix.
        - mat2 (matrix-like of scalar or vector): The right matrix.

    Returns:
        - mat (matrix-like of float or vector): The matrix result.
    """
    return tuple(mult_rmat_vec(mat2, row) for row in mat1)
