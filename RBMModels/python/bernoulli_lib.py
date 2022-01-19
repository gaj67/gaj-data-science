"""
This library provides a collection of methods for dealing with binary
inputs and outputs for a Bernoulli Restricted Boltzmann Machine.

The basic RBM with input vector x and output vector y takes the form:

    p(x,y) = exp{-E(x,y)} / Z_XY

for energy function

    E(x,y) = -[ a^T x + b^T y + x^T W y ]

and normalising partition function Z_XY.

For Bernoulli inputs and outputs, x and y become vectors of binary values.
Note that the partition function remains intractable in general, but the
conditional distributions become tractable. In particular:

    P(x_i = 1 | y) = logistic( a_i + [W y]_i )

and

    P(y_j = 1 | x) = logistic( b_j + [x^T W]_j )

The marginal distribution, which is also still intractable, is then

    p(x) = exp{-F(x)} / Z_X

with free energy

    F(x) = -[ a^T x + sum_{j} log( 1 + exp{ b_j + [x^T W]_j } ) ]
"""

import numpy as np
from scipy.special import expit as logistic
import vector_lib as vlib


def output_probs(b, W, X):
    """
    Computes the conditional probabilities [[q_ij]] that the j-th output unit
    has value 1 (for j=1,2,...,H), given the i-th input vector (i=1,2,...,N).
    The model predicts:

        q_ij = P(y_j = 1 | x_i) = logistic( b_j + sum_{k=1}^F x_ik W_kj )

    Inputs:
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix [x_i] of input cases.
    Returns:
        - Q (array): The N x H matrix of output probabilities.
    """
    return logistic(np.matmul(X, W) + b)


def grad_output_probs(X, Q):
    """
    Computes the mean-data gradients, with respect to the model parameters, of
    the Bernoulli output probabilities. The gradients take the form:

        mean(grad_theta(Q)) = [ 1/N sum_{i=1}^N d q_ij/d theta ]

    Inputs:
        - X (array): The N x F matrix of input cases.
        - Q (array): The N x H matrix of output probabilities.
    Returns:
        - grad_b (array): The size-H array of gradients with respect to 'b'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    # NB: d logistic(z)/d z = logistic(z) [1 - logistic(z)]
    grad_z = Q * (1 - Q)
    grad_b = np.mean(grad_z, axis=0)
    grad_W = np.matmul(X.T, grad_z) / X.shape[0]
    return grad_b, grad_W


def input_probs(a, W, Y):
    """
    Computes the conditional probabilities [[p_ij]] that the j-th input unit
    has value 1 (for j=1,2,...,F), given the i-th output vector (i=1,2,...,N).
    The model predicts:

        p_ij = P(x_j = 1 | y_i) = logistic( a_j + sum_{k=1}^H y_ik W_jk )

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - W (array): The F x H matrix of interaction weights.
        - Y (array): The N x H matrix of output cases.
    Returns:
        - P (array): The N x F matrix of input probabilities.
    """
    return logistic(np.matmul(Y, W.T) + a)


def grad_input_probs(Y, P):
    """
    Computes the mean-data gradients, with respect to the model parameters, of
    the Bernoulli input probabilities. The gradients take the form:

        mean(grad_theta(P)) = [ 1/N sum_{i=1}^N d p_ij/d theta ]

    Inputs:
        - Y (array): The N x H matrix of output cases.
        - P (array): The N x F matrix of input probabilities.
    Returns:
        - grad_a (array): The size-F array of gradients with respect to 'a'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    grad_z = P * (1 - P)
    grad_a = np.mean(grad_z, axis=0)
    grad_W = np.matmul(grad_z.T, Y) / Y.shape[0]
    return grad_a, grad_W


def free_energies(a, b, W, X):
    """
    Computes the free energies of the given input vectors.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - F (array): The size-N free energies.
    """
    term1 = np.matmul(X, a)
    exponent = np.matmul(X, W) + b
    term2 = np.sum(np.log(1 + np.exp(exponent)), axis=1)
    return -(term1 + term2)


def mean_free_energy(a, b, W, X):
    """
    Computes the mean free energy of the given input vectors.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - F (float): The mean free energy.
    """
    return np.mean(free_energies(a, b, W, X))


def grad_gibbs_sampling(a, b, W, X):
    """
    Computes the Gibbs sampling approximations of the gradients, with respect
    to the model parameters, of the mean logarithm of the joint probability
    of the input data.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - grad_a (array): The size-F array of gradients with respect to 'a'.
        - grad_b (array): The size-H array of gradients with respect to 'b'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    N = X.shape[0]
    Q = output_probs(b, W, X)  # E_{y|x}[ y ]
    Y = bernoulli_sample(Q)
    P = input_probs(a, W, Y)  # E_{x'|y}[ x' ]
    Xd = bernoulli_sample(P)
    Qd = output_probs(b, W, Xd)  # E_{y'|x'}[ y' ]
    grad_a = np.mean(X, axis=0) - np.mean(Xd, axis=0)
    grad_b = np.mean(Y, axis=0) - np.mean(Qd, axis=0)
    grad_W = (np.matmul(X.T, Y) - np.matmul(Xd.T, Qd)) / N
    return grad_a, grad_b, grad_W


def grad_hinton_gibbs(a, b, W, X):
    """
    Computes the Hinton modification of the Gibbs sampling approximations of
    the gradients, with respect to the model parameters, of the mean logarithm
    of the joint probability of the input data.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - grad_a (array): The size-F array of gradients with respect to 'a'.
        - grad_b (array): The size-H array of gradients with respect to 'b'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    N = X.shape[0]
    Q = output_probs(b, W, X)  # E_{y|x}[ y ]
    Y = bernoulli_sample(Q)
    P = input_probs(a, W, Y)  # E_{x'|y}[ x' ]
    Qd = output_probs(b, W, P)  # E_{x'|y}[ E_{y'|x'}[ y' ] ]
    grad_a = np.mean(X, axis=0) - np.mean(P, axis=0)
    grad_b = np.mean(Y, axis=0) - np.mean(Qd, axis=0)
    grad_W = (np.matmul(X.T, Y) - np.matmul(P.T, Qd)) / N
    return grad_a, grad_b, grad_W


def grad_mean_field(a, b, W, X):
    """
    Computes the mean field approximations of the gradients, with respect to
    the model parameters, of the mean logarithm of the joint probability of
    the input data.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - grad_a (array): The size-F array of gradients with respect to 'a'.
        - grad_b (array): The size-H array of gradients with respect to 'b'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    N = X.shape[0]
    Q = output_probs(b, W, X)  # E_{y|x}[ y ]
    P = input_probs(a, W, Q)  # E_{y|x}[ E_{x'|y}[ x' ] ]
    Qd = output_probs(b, W, P)  # E_{y|x}[ E_{x'|y}[ E_{y'|x'}[ y' ] ] ]
    grad_a = np.mean(X, axis=0) - np.mean(P, axis=0)
    grad_b = np.mean(Q, axis=0) - np.mean(Qd, axis=0)
    grad_W = (np.matmul(X.T, Q) - np.matmul(P.T, Qd)) / N
    return grad_a, grad_b, grad_W


def reconstruction_probs(a, b, W, X):
    """
    Computes the mean field approximate probabilities [[p_ij]] that the j-th
    input unit has value 1 (for j=1,2,...,H), given the i-th input vector
    (for i=1,2,...,N).
    The model predicts:

        q_ik = P(y_k = 1 | x_i) = logistic( b_k + sum_{j=1}^F x_ij W_jk )
        p_ij = P(x_j = 1 | q_i) = logistic( a_j + sum_{k=1}^H q_ik W_jk )

    In other words, the usual step of sampling a Bernoulli vector from the
    output probabilities is skipped. The output probabilities themselves are
    'plugged' into the function to compute approximate input probabilities.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - Q (array): The N x H matrix of output probabilities.
        - P (array): The N x F matrix of reconstructed input probabilities.
    """
    Q = output_probs(b, W, X)
    P = input_probs(a, W, Q)
    return Q, P


def reconstruction_scores(X, P):
    """
    Computes the logarithm of the approximate probabilities of the
    input data. The scores are given by:

        log prod_{j=1}^F [ p_ij^x_ij * (1 - p_ij)^(1-x_ij) ]

    Inputs:
        - X (array): The N x F matrix of input cases.
        - P (array): The N x F matrix of reconstructed input probabilities.
    Returns:
        - scores (array): The log approximate probabilities.
    """
    S = X * np.log(P + 1e-30) + (1 - X) * np.log(1 - P + 1e-30)
    return np.sum(S, axis=1)


def mean_reconstruction_score(X, P):
    """
    Computes the mean logarithm of the joint approximate probability of the
    input data. The score is given by:

        1/N sum_{i=1}^N log prod_{j=1}^F [ p_ij^x_ij * (1 - p_ij)^(1-x_ij) ]

    Inputs:
        - X (array): The N x F matrix of input cases.
        - P (array): The N x F matrix of reconstructed input probabilities.
    Returns:
        - score (float): The mean log approximate probability.
    """
    return np.mean(reconstruction_scores(X, P))


def grad_reconstruction_score(W, X, Q, P):
    """
    Computes the gradients, with respect to the model parameters, of the
    mean logarithm of the joint approximate probability of the
    input data.

    Inputs:
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
        - Q (array): The N x H matrix of output probabilities.
        - P (array): The N x F matrix of reconstructed input probabilities.
    Returns:
        - grad_a (array): The size-F array of gradients with respect to 'a'.
        - grad_b (array): The size-H array of gradients with respect to 'b'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    # P = logistic(z), z = a + W Q
    # Note: dP/dz = P * (1 - P)
    # S = X * log(P) + (1 - X) * log(1 - P)
    # => dS/dP = X / P - (1 - X) / (1 - P)
    # => dS/dz = dS/dP dP/dz = X - P
    dS_dz = X - P  # N x F
    # dS/da = dS/dz dz/da, but dz/da = 1
    grad_a = np.mean(dS_dz, axis=0)
    # Q = logistic(t), t = b + X W
    dQ_dt = Q * (1 - Q)  # N x H
    # dS/db = dS/dz dz/db; dz/db = W dQ/db; dQ/db = dQ/dt dt/db but dt/db = 1
    grad_b = np.mean(dQ_dt * np.matmul(dS_dz, W), axis=0)
    # dS/dW = dS/dz {Q + W dQ/dW}; dQ/dW = dQ/dt dt/dW = dQ/dt X
    term1 = np.matmul(dS_dz.T, Q)
    term2 = W * np.matmul((dS_dz * X).T, dQ_dt)
    grad_W = (term1 + term2) / X.shape[0]
    return grad_a, grad_b, grad_W


def mean_reconstruction_error(X, P):
    """
    Computes the mean square error of reconstructing the inputs using
    approximate probabilities. The MSE is given by:

        1/N sum_{i=1}^N sum_{j=1}^F ( x_ij - p_ij )^2

    Inputs:
        - X (array): The N x F matrix of input cases.
        - P (array): The N x F matrix of reconstructed input probabilities.
    Returns:
        - mse (float): The mean square error.
    """
    return np.mean(np.sum((X - P) ** 2, axis=1))


def grad_reconstruction_error(W, X, Q, P):
    """
    Computes the gradients, with respect to the model parameters, of the
    mean square error of reconstructing the input data.

    Inputs:
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
        - Q (array): The N x H matrix of output probabilities.
        - P (array): The N x F matrix of reconstructed input probabilities.
    Returns:
        - grad_a (array): The size-F array of gradients with respect to 'a'.
        - grad_b (array): The size-H array of gradients with respect to 'b'.
        - grad_W (array): The F x H matrix of gradients with respect to 'W'.
    """
    # P = logistic(z) => dP/dz = P * (1 - P)
    # S = (X - P)^2 => dS/dz = dS/dP dP/dz = -2 (X - P) * P * (1 - P)
    dS_dz = -2 * (X - P) * P * (1 - P)
    # z = a + W Q => dS/da = dS/dz dz/da = dS/dz
    grad_a = np.mean(dS_dz, axis=0)
    # Q = logistic(t) => dQ/dt = Q * (1 - Q)
    dQ_dt = Q * (1 - Q)  # N x H
    # t = b + X W = > dQ/db = dQ/dt dt/db = dQ/dt
    # => dS/db = dS/dz dz/db = dS/dz W dQ/db
    grad_b = np.mean(dQ_dt * np.matmul(dS_dz, W), axis=0)
    # dz/dW = Q + W dQ/dW, dQ/dW = dQ/dt dt/dW = dQ/dt X
    # => dS/dW = dS/dz {Q + W dQ/dt X}
    term1 = np.matmul(dS_dz.T, Q)
    term2 = W * np.matmul((dS_dz * X).T, dQ_dt)
    grad_W = (term1 + term2) / X.shape[0]
    return grad_a, grad_b, grad_W


def sequential_reconstruction_probs(a, b, W, X):
    """
    Computes the NADE mean-field probabilities [[p_ij]] that the j-th input
    unit has value 1 (for j=1,2,...,F), for the i-th input vector (i=1,2,...,N).
    The model predicts:

        q_ijk = P(y_k = 1 | x_i) = logistic( b_k + sum_{m=1}^{j-1} x_im W_mk )
        p_ij = P(x_ij = 1 | x_i) = logistic( a_j + sum_{k=1}^{H} W_jk q_ijk )

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - P (array): The N x F array of probabilities.
    """
    N, F = X.shape
    H = len(b)
    P = np.zeros((N, F))
    Z = np.tile(b, (N, 1))  # N x H array of partial sums
    for j in range(F):
        Q = logistic(Z)
        P[:, j] = logistic(np.matmul(Q, W[j, :]) + a[j])
        Z += np.outer(X[:, j], W[j, :])
    return P


def grad_sequential_reconstruction_probs(a, b, W, X):
    """
    Computes the exact gradients of the mean log of the NADE mean-field
    probabilities.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - grad_a (array): The size-F vector gradient for 'a'.
        - grad_b (array): The size-H vector gradient for 'b'.
        - grad_W (array): The F x H matrix gradient for 'W'.
    """
    N, F = X.shape
    H = len(b)
    grad_a = np.zeros(F)
    grad_W = np.zeros((F, H))
    sum_B = np.zeros((N, H))  # N x H array of partial sums
    Z = np.matmul(X, W) + b  # N x H array of partial sums
    for j in range(F - 1, -1, -1):
        x_j = X[:, j]
        w_j = W[j, :]
        Z -= np.outer(x_j, w_j)
        Y_bar = logistic(Z)  # p(y=1 | x_1,x_2,...,x_{j-1})
        x_bar_j = logistic(np.matmul(Y_bar, w_j) + a[j])  # p(x_j = 1 | y)
        d_j = x_j - x_bar_j
        grad_a[j] = np.mean(d_j)
        grad_W[j, :] = (np.matmul(d_j, Y_bar) + np.matmul(x_j, sum_B)) / N
        B = vlib.multiply_columns(Y_bar * (1 - Y_bar) * w_j, d_j)
        sum_B += B
    grad_b = np.mean(B, axis=0)
    return grad_a, grad_b, grad_W


def approx_grad_sequential_reconstruction_probs(a, b, W, X):
    """
    Computes approximate gradients of the mean log of the NADE mean-field
    probabilities, by extending the the mean field approximation gradients
    of the standard RBM model.

    Inputs:
        - a (array): The F-sized vector of visible weights.
        - b (array): The H-sized vector of hidden weights.
        - W (array): The F x H matrix of interaction weights.
        - X (array): The N x F matrix of input cases.
    Returns:
        - grad_a (array): The size-F vector gradient for 'a'.
        - grad_b (array): The size-H vector gradient for 'b'.
        - grad_W (array): The F x H matrix gradient for 'W'.
    """
    N, F = X.shape
    H = len(b)
    grad_a = np.zeros(F)
    grad_b = np.zeros(H)
    grad_W = np.zeros((F, H))
    Z = np.tile(b, (N, 1))  # N x H array of partial sums
    for j in range(F):
        Q = logistic(Z)
        P_j = logistic(np.matmul(Q, W[j, :]) + a[j])
        grad_a[j] = np.mean(X[:, j]) - np.mean(P_j)
        Z_hat = Z + np.outer(P_j, W[j, :])
        Q_hat = logistic(Z_hat)
        grad_b += np.mean(Q, axis=0) - np.mean(Q_hat, axis=0)
        grad_W[j, :] = (np.matmul(X[:, j], Q) - np.matmul(P_j, Q_hat)) / N
        Z += np.outer(X[:, j], W[j, :])
    return grad_a, grad_b, grad_W
