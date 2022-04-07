"""
Implements some stand-alone methods for vector and matrix manipulation.
"""

import numpy as np
from scipy.special import expit as logistic


def is_matrix(X):
    """
    Determines whether or not the input is a 2D matrix.

    Input:
        - X: The object to test.
    Returns:
        res (bool): A value of True (else False) if the input is a matrix.
    """
    return isinstance(X, np.ndarray) and len(X.shape) == 2


def random_tensor(*dimensions):
    """
    Creates an array with random elements in the range (-scale, +scale), where
    the scale is the inverse of the product of the dimensions.
    """
    return (2.0 * np.random.rand(*dimensions) - 1) / np.prod(dimensions)


def orthogonalise_columns(X):
    """
    Performs in-place orthogonalisation of the matrix columns.

    Input:
        - X (array): An N x M matrix.
    Returns:
        - X (array): The N x M modified matrix.
    """
    M = X.shape[1]
    sq_norms = np.zeros(M)
    for i in range(M):
        v_i = X[:, i]
        for j in range(0, i):
            u_j = X[:, j]
            v_i -= u_j * np.dot(v_i, u_j) * sq_norms[j]
        sq_len = np.dot(v_i, v_i)
        sq_norms[i] = 1 if sq_len < 1e-16 else 1 / sq_len


def multiply_columns(A, v, inplace=True):
    """
    Performs element-wise multiplication of the matrix columns by the
    given vector.

    Inputs:
        - A (array): An N x M matrix.
        - v (array): A size-N vector.
        - inplace (bool, default=True): Indicates whether (True) or not
            (False) to directly modify the input matrix.
    Returns:
        - B (array): The N x M (possibly modified) matrix.
    """
    M = A.shape[1]
    if inplace:
        for j in range(M):
            A[:, j] *= v
        return A
    else:
        B = np.zeros(A.shape)
        for j in range(M):
            B[:, j] = A[:, j] * v
        return B


def vector_softmax(v):
    """
    Exponentiates the elements of the given vector, and normalises the
    result to sum to unity.

    Input:
        - v (array): A size-N vector.
    Returns:
        - s (array): The size-N soft-max vector.
    """
    # Guard against values with very high magnitude
    s = np.exp(v - np.max(v)) + 1e-30
    s /= sum(s)
    return s


def row_softmax(A):
    """
    Exponentiates the elements of the given matrix, and normalises the
    resulting rows to sum to unity.

    Input:
        - A (array) - An N x M matrix.
    Returns:
        - S (array) - The N x M (row-stochastic) soft-max matrix.
    """
    N, M = A.shape
    Z = np.zeros((N, M))
    if N < M:
        # Work row-wise
        for i in range(N):
            Z[i, :] = vector_softmax(A[i, :])
    else:
        # Work column-wise
        # Guard against values with very high magnitude
        row_max = np.max(A, axis=1)
        Z_sum = np.zeros(N)
        for j in range(M):
            Z[:, j] = col = np.exp(A[:, j] - row_max) + 1e-30
            Z_sum += col
        for j in range(M):
            Z[:, j] /= Z_sum
    return Z


def square_errors(X, Y):
    """
    Computes the squared errors between the row vectors X = [x_i]
    and Y = [y_i].

    Inputs:
        - X (array): The N x M matrix of observed values.
        - Y (array): The N x M matrix of expected values.
    Returns:
        - errs (array): The N-sized vector of squared errors.
    """
    return np.sum((X - Y) ** 2, axis=1)


def one_hot_means(X, M=None):
    """
    Computes the proportion of times each bit is set to 1. This is equivalent
    to the empirical expectation or probability.

    Input:
        - X (array): Either an N-sized vector of indices, or an N x M matrix
            of one-hot row vectors.
        - M (int, optional): The number of bits, if known.
    Returns:
        - probs (array): The bit proportions.
    """
    if is_matrix(X):
        return np.mean(X, axis=0)
    if M is None:
        M = 1 + max(X)
    return np.array([np.mean(X == i) for i in range(M)])


def multiply_one_hot(X, Y, B=None):
    """
    Computes R = X * Y for matrix Y of one-hot row vectors.

    Input:
        - X (array): An M x N matrix.
        - Y (array): An N x B matrix of one-hot row vectors, or a size-N vector
            of indices in range(0, B).
        - B (int, optional): The number of bits, if known.
    Returns:
        - R (array): The M x B matrix product.
    """
    if is_matrix(Y):
        return np.matmul(X, Y)
    if B is None:
        B = 1 + max(Y)
    M, N = X.shape
    R = np.zeros((M, B), dtype=X.dtype)
    for i in range(N):
        R[:, Y[i]] += X[:, i]
    return R


def one_hot_multiply(Y, X):
    """
    Computes R = Y * X for matrix Y of one-hot row vectors.

    Input:
        - Y (array): An N x B matrix of one-hot row vectors, or a size-N vector
            of indices in range(0, B).
        - X (array): A B x M matrix.
    Returns:
        - R (array): The N x M matrix product.
    """
    if is_matrix(Y):
        return np.matmul(Y, X)
    return X[Y, :]
