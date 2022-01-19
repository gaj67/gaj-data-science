"""
Implements some stand-alone methods for vector and matrix manipulation.
"""

import numpy as np


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


def multiply_columns(A, v):
    """
    Performs in-place, element-wise multiplication of the matrix columns by the
    given vector.

    Inputs:
        - A (array): An N x M matrix.
        - v (array): A size-N vector.
    Returns:
        - A (array): The N x M modified matrix.
    """
    M = A.shape[1]
    for j in range(M):
        A[:, j] *= v
    return A


def binary_sample(probs):
    """
    Stochastically assigns 1 (else 0) for each element, with the given element's
    Bernoulli probability.

    Input:
        - probs (array): An arbitrarily-sized tensor of independent probabilities.
    Returns:
        - res (array): The resulting binary tensor.
    """
    return np.asarray(np.floor(probs + np.random.rand(*probs.shape)), dtype=int)


def binary_decision(probs):
    """
    Deterministically assigns 1 (else 0) to each element, if the element's
    Bernoulli probability exceeds 0.5.

    Input:
        - probs (array): An arbitrarily-sized tensor of independent probabilities.
    Returns:
        - res (array): The resulting binary tensor.
    """
    return np.asarray(probs > 0.5, dtype=int)


def binary_vector(value, N=None):
    """
    Converts the decimal value into a size-N vector of bits, using zero-padding
    or truncation (both on the left) as necessary.

    Inputs:
        - value (int): The decimal value.
        - N (int, optional): The size of the binary vector.
    Returns:
        - vec (array): The size-N binary vector.
    """
    b = bin(value)[2:]
    if N is not None:
        b = b[-N:]
        if len(b) < N:
            b = "0" * (N - len(b)) + b
    return np.asarray(list(int(bit) for bit in b), dtype=int)


def binary_matrix(values, N=None):
    """
    Converts the decimal values into an M x N matrix of bits, using zero-padding
    or truncation (both on the left) as necessary.

    Inputs:
        - values (iterable of int): The size-M collection of decimal values.
        - N (int, optional): The size of each binary row vector.
    Returns:
        - mat (array): The M x N binary vector.
    """
    if N is None:
        N = len(bin(max(values))) - 2
    return np.asarray([binary_vector(v, N) for v in values])
