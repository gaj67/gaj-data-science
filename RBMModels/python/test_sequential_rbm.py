"""
Tests the various implementations of Restricted Boltzmann Machines.
"""

import numpy as np
import bernoulli_lib as blib
from bernoulli_rbm import BernoulliRBM, SequentialBernoulliRBM
import vector_lib as vlib
from scipy.special import expit as logistic


def plural_str(d):
    return "s" if d > 1 else ""


def gen_params(F, H):
    W = vlib.random_tensor(F, H)
    vlib.orthogonalise_columns(W)
    a = -0.5 * np.sum(W, axis=1)
    b = -0.5 * np.sum(W, axis=0)
    return a, b, W


if "__main__" == __name__:
    # Incrementally test models
    print("Experiment 1: compare standard versus sequential model")
    for num_bits in range(1, 4):
        print("Training with %d bit%s..." % (num_bits, plural_str(num_bits)))
        X = vlib.binary_matrix(range(2 ** num_bits))
        a, b, W = gen_params(num_bits, num_bits)
        print("Training standard RBM...")
        std_model = BernoulliRBM(
            params=(a.copy(), b.copy(), W.copy()),
            n_iter=25,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        print(std_model._report_card(X, 0))
        for i in range(4):
            std_model.fit(X)
            print(std_model._report_card(X, (i + 1) * std_model.n_iter))
        Q, P = blib.reconstruction_probs(*std_model._params, X)
        print(P)
        print("Training sequential RBM...")
        seq_model = SequentialBernoulliRBM(
            params=(a.copy(), b.copy(), W.copy()),
            n_iter=25,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        print(seq_model._report_card(X, 0))
        for i in range(4):
            seq_model.fit(X)
            print(seq_model._report_card(X, (i + 1) * seq_model.n_iter))
        P = blib.sequential_reconstruction_probs(*seq_model._params, X)
        print(P)
    # Done

    # Test alternating sequence
    print("Experiment 2: alternating sequences")
    X = np.array([(1, 0, 1, 0), (0, 1, 0, 1)])
    for num_hidden in range(1, 5):
        print("Using %d output bit%s" % (num_hidden, plural_str(num_hidden)))
        a, b, W = gen_params(X.shape[1], num_hidden)
        print("Training standard model...")
        std_model = BernoulliRBM(
            params=(a.copy(), b.copy(), W.copy()),
            n_iter=300,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        std_model.fit(X)
        print(std_model._report_card(X))
        Q, P = blib.reconstruction_probs(*std_model._params, X)
        print(P)
        print("Training sequential model...")
        seq_model = SequentialBernoulliRBM(
            params=(a.copy(), b.copy(), W.copy()),
            n_iter=300,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        seq_model.fit(X)
        print(seq_model._report_card(X))
        P = blib.sequential_reconstruction_probs(*seq_model._params, X)
        print(P)
