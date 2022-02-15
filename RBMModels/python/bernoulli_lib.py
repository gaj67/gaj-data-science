"""
This library provides a collection of methods for dealing with binary
inputs and outputs for Bernoulli Restricted Boltzmann Machines (RBMs)
and Classifiers (RBCs).

The standard Bernoulli RBM (hereafter referred to as RBM-std) is a
discrete exponential model with joint probability p(x,z) taking the form

                   exp{ a^T x + b^T z + x^T W z }
    ------------------------------------------------
    sum_x' sum_z' exp{ a^T x' + b^T z' + x'^T W z' }

for input layer x in {0,1}^F and hidden layer z in {0,1}^H.

Notionally, the undirected model x -- z corresponds to both directed models
x -> z (forward-logistic) and x <- z (backward-logistic). In terms of neural
networks, 'z' has a linear layer with logsitic sigmoid activations.

A detailed derivation of RBMs is given in the text document
"notebooks/rbm_models.ipynb".


The standard Bernoulli RBC (hereafter referred to as RBC-std) is a discrete
RBM with joint probability p(x,y,z) taking the form

                            exp{ a^T x + b^T z + x^T W z + c^T y + z^T U y }
    ----------------------------------------------------------------------------
    sum_x' sum_y' sum_z' exp{ a^T x' + b^T z' + x'^T W z' + c^T y' + z'^T U y' }

with an additional output layer y in {0,1}^C restricted to one-hot vectors.

The undirected model x -- z -- y now corresponds to the directed models
x -> z -> (hidden-softmax), x -> z <- y (bi-logistic), and
(backward-logistic) x <- z -> y (forward-softmax). In terms of neural
networks, 'y' has a linear layer with soft-max activations.

A detailed derivation of RBCs is given in the text document
"notebooks/rbm_classifiers.ipynb", which also refers to the mechanics of the
more general Boltzmann machine, which is derived in detail in the text document
"notebooks/discrete_boltzmann.ipynb".


As a special case, the Bernoulli logistic RBC (referred to as RBC-logistic)
is a Bernoulli RBC without the hidden layer. This has joint probability p(x,y)
taking the form

                   exp{ a^T x + c^T y + x^T U y }
    ------------------------------------------------
    sum_x' sum_y' exp{ a^T x' + c^T y' + x'^T U y' }

The undirected model x -- y corresponds to the directed models
x -> y (forward-softmax) and x <- y (backward-logistic).


Nomenclature of constants:
    - N: The number of cases in the training data-set.
    - F: The number of features in the binary vector x.
    - C: The number of classes in the one-hot vector y.
    - H: The number of hidden units in the binary vector z.

Nomenclature of parameters:
    - a: The size-F vector of input biases.
    - b: The size-H vector of hidden biases.
    - W: The F x H matrix of input/hidden weights.
    - c: The size-C vector of output biases.
    - U: The H x C matrix of hidden/output weights.

Nomenclature of variables:
    - d: Indexes the N training data cases.
    - i: Indexes the F input feature units.
    - j: Indexes the C output classes.
    - k: Indexes the H hidden units.
    - X_bar: The conditional expectation E[x | z].
    - Y_bar: The conditional expectation E[y | z].
    - Y_tilde: The conditional expectation E[y | x] (RBC).
    - Z_bar: The conditional expectation E[z | x] (RBM) or E[z | x, y] (RBC).

Nomenclature of arrays:
    - vectors denoted by lowercase, e.g. a or x or y.
    - matrices denoted by upper case, e.g. W or X or Y.
    - scalar and vector elements denoted either by upper case or lower case:
        - _i indicates the i-th element of a vector, e.g. x_i.
        - _ik indicates the i-th row and k-th column of a matrix,
          e.g. W_ik or w_ik.
        - _:j indicates the j-th column of a matrix, e.g. U_:j.
        - _i: indicates the i-th row of a matrix, e.g. W_i:.
"""

import numpy as np
import vector_lib as vlib


# Allow for x to potentially be a Markov sequence
_forward_logistic = vlib.logistic


def forward_logistic(X, W, b):
    """
    For the RBM-std or partial RBC-std models

        x -[weights W]- z (bias b)

    this method computes

        E[z_k | x_d] = p(z_k=1 | x_d) = logistic( x_d^T W_:k + b_k )

    Inputs:
        - X (array): An N x F matrix of input row vectors.
        - W (array): The F x H matrix of input/hidden weights.
        - b (array): The size-H vector of hidden biases.
    Returns:
        - Z_bar (array): The N x H matrix of logistic expectations.
    """
    return _forward_logistic(np.matmul(X, W) + b)


def backward_logistic(Z, a, W):
    """
    For the RBM-std or partial RBC-std models

        x (bias a) -[weights W]- z

    this method computes

        E[x_i | z_d] = p(x_i=1 | z_d) = logistic( W_i: z_d + a_i )

    Inputs:
        - Z (array): An N x H matrix of 'hidden' row vectors.
        - a (array): The size-F vector of input biases.
        - W (array): The F x H matrix of input/hidden weights.
    Returns:
        - X_bar (array): The N x F matrix of logistic expectations.
    """
    return vlib.logistic(np.matmul(Z, W.T) + a)


# Allow for x to potentially be a Markov sequence
def _bi_logistic(b_plus_XW, Y, U):
    return vlib.logistic(b_plus_XW + vlib.one_hot_multiply(Y, U.T))


def bi_logistic(X, Y, W, b, U):
    """
    For the RBC-std model

        x -[weights W] - z (bias b) -[weights U]- y

    this method computes

        p(z_k = 1 | x_d, y_d) = logistic( x_d W_:k + U_k: y_d + b_k )

    Inputs:
        - X (array): The N x F matrix of input row vectors.
        - Y (array): Either an N x C matrix of output vectors, or a
            size-N vector of output indices.
        - W (array): The F x H matrix of input/hidden weights.
        - b (array): The size-H vector of hidden biases.
        - U (array): The H x C matrix of hidden/output weights.
    Returns:
        - Z_bar (array): The N x H matrix of hidden expectations.
    """
    return _bi_logistic(np.matmul(X, W) + b, Y, U)


# Allow for x to potentially be a Markov sequence
_forward_softmax = vlib.row_softmax


def forward_softmax(Z, U, c):
    """
    For the partial RBC-std model

        z -[weights U]- y (bias c)

    this method computes

        p(y_j = 1 | z_d) = softmax( z_d U_:j + c_j )

    Inputs:
        - Z (array): An N x H matrix of 'hidden' row vectors.
        - U (array): The H x C matrix of hidden/output weights.
        - c (array): The size-C vector of output biases.
    Returns:
        - Y_bar (array): The N x C matrix of output probabilities, with unit
            row sums.
    """
    return _forward_softmax(np.matmul(Z, U) + c)


# Allow for x to potentially be a Markov sequence
def _hidden_softmax(b_plus_XW, U, c):
    N = b_plus_XW.shape[0]
    C = U.shape[1]
    ln_P = np.zeros((N, C))
    exp_bXW = np.exp(b_plus_XW)  # N x H
    exp_U = np.exp(U)  # H x C
    for j in range(C):
        ln_P[:, j] = np.sum(np.log(exp_bXW * exp_U[:, j] + 1), axis=1) + c[j]
    return vlib.row_softmax(ln_P)


def hidden_softmax(X, W, b, U, c):
    """
    For the RBC-std model

        x -[weights W]- z (bias b) -[weights U]- y (bias c)

    this method computes

        E[y_j|x_d] = p(y_j=1|x_d) = softmax(...x_d...summed over z...)

    where 'z' is a binary vector.

    Inputs:
        - X (array): An N x F matrix of input cases.
        - W (array): The F x H matrix of input/hidden weights.
        - b (array): The size-H vector of hidden biases.
        - U (array): The H x C matrix of hidden/output weights.
        - c (array): The size-C vector of output biases.
    Returns:
        - Y_tilde (array): The N x C matrix of output probabilities, with unit
            row sums.
    """
    return _hidden_softmax(np.matmul(X, W) + b, U, c)


# Allow for x to potentially be a Markov sequence
def _hidden_logistic(b_plus_XW, U, c):
    N = b_plus_XW.shape[0]
    C = U.shape[1]
    ln_P = np.zeros((N, C))
    exp_bXW = np.exp(b_plus_XW)  # N x H
    exp_U = np.exp(U)  # H x C
    for j in range(C):
        ln_P[:, j] = np.sum(np.log(exp_bXW * exp_U[:, j] + 1), axis=1) + c[j]
    return vlib.logistic(ln_P)


def hidden_logistic(X, W, b, U, c):
    """
    For the 3-layer RBM model

        x -[weights W]- z (bias b) -[weights U]- y (bias c)

    with binary output 'y', this method computes

        E[y_j|x_d] = p(y_j=1|x_d) = logistic(...x_d...summed over z...)

    where 'z' is a binary vector.

    Inputs:
        - X (array): An N x F matrix of input cases.
        - W (array): The F x H matrix of input/hidden weights.
        - b (array): The size-H vector of hidden biases.
        - U (array): The H x C matrix of hidden/output weights.
        - c (array): The size-C vector of output biases.
    Returns:
        - Y_tilde (array): The N x C matrix of output probabilities.
    """
    return _hidden_logistic(np.matmul(X, W) + b, U, c)


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


def binary_scores(X, P):
    """
    Computes the log-likelihoods of the binary row vectors, X = [x_i],
    given the probabilities, P = [p_i], of each bit being independently
    set to 1.

    The scores are given by:

        log p(x_i) = log prod_j [ p_ij^x_ij * (1 - p_ij)^(1-x_ij) ]

    Note that if X is None (i.e. every x_i is unknown), then

        E[log p(x_i)] = sum_{x} p(x) log p(x)

    is computed instead.

    Inputs:
        - X (array): The N x M matrix of binary input vectors.
        - P (array): The N x M matrix of bit probabilities.
    Returns:
        - scores (array): The size-N vector of log-likelihoods.
    """
    if X is None:
        X = P  # force expected values
    S = X * np.log(P + 1e-30) + (1 - X) * np.log(1 - P + 1e-30)
    return np.sum(S, axis=1)


def binary_errors(X, P):
    """
    Computes the number of bit-wise errors made by deterministically
    reconstructing the binary row vectors, X = [x_i], from the given
    probabilities, P = [p_i], of each bit being independently
    set to 1.

    The scores are given by:

        e(x_i) = sum_j abs( x_ij - decide(p_ij) )

    Inputs:
        - X (array): The N x M matrix of binary input vectors.
        - P (array): The N x M matrix of bit probabilities.
    Returns:
        - scores (array): The size-N vector of bit errors.
    """
    return np.sum(np.abs(X - binary_decision(P)), axis=1)


def one_hot_scores(X, P):
    """
    Computes the log-likelihoods of the observed cases, X=[x_i], given the
    (dependent) probabilities, P = [[p_ij]], that x_i belongs to class j, i.e.
    the corresponding one-hot vector has a 1 at the j-th bit (with all other
    bits being 0).

    If X is a matrix of one-hot row vectors, then the scores are given by:

        log p(x_i) = sum_{j} x_ij log p_ij

    Otherwise, if X is a vector of class indices, then the scores are given by:

        log p(x_i) = log p_{i, x_i}

    Note that if X is None (i.e. every x_i is unknown), then the expected value

        E[log p(x_i)] = sum_{x} p(x) log p(x)

    is computed instead.

    Inputs:
        - X (array): Either an N-sized vector of indices, or an N x M matrix
            of one-hot row vectors.
        - P (array): The N x M matrix of class probabilities.
    Returns:
        - scores (array): The N-sized vector of log-likelihoods.
    """
    log_P = np.log(P + 1e-30)
    if X is None:
        X = P  # force expected values
    if vlib.is_matrix(X):
        # matrix
        return np.sum(X * log_P, axis=1)
    else:
        # assume vector
        return log_P[range(len(X)), X]
