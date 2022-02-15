"""
Tests the various implementations of Restricted Boltzmann Machines.
"""

import numpy as np
import bernoulli_lib as blib
import vector_lib as vlib
from bernoulli_rbm import StandardBernoulliRBM, SequentialBernoulliRBM


def plural_str(d):
    return "s" if d > 1 else ""


def gen_params(F, H):
    W = vlib.random_tensor(F, H)
    vlib.orthogonalise_columns(W)
    a = -0.5 * np.sum(W, axis=1)
    b = -0.5 * np.sum(W, axis=0)
    return a, W, b


if "__main__" == __name__:
    # Incrementally test models
    print("Experiment 1: compare standard versus sequential model")
    for num_bits in range(1, 4):
        print("Training with %d bit%s..." % (num_bits, plural_str(num_bits)))
        X = blib.binary_matrix(range(2 ** num_bits))
        a, W, b = gen_params(num_bits, num_bits)
        print("Training standard RBM...")
        std_model = StandardBernoulliRBM(
            params=(a.copy(), W.copy(), b.copy()),
            n_iter=100,
            n_report=25,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        std_model.fit(X)
        print("Training sequential RBM...")
        seq_model = SequentialBernoulliRBM(
            params=(a.copy(), W.copy(), b.copy()),
            n_iter=100,
            n_report=25,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        seq_model.fit(X)
    # Done

    # Test alternating sequence
    print("Experiment 2: alternating sequences")
    X = np.array([(1, 0, 1, 0), (0, 1, 0, 1)])
    for num_hidden in range(1, 5):
        print("Using %d output bit%s" % (num_hidden, plural_str(num_hidden)))
        a, W, b = gen_params(X.shape[1], num_hidden)
        print("Training standard model...")
        std_model = StandardBernoulliRBM(
            params=(a.copy(), W.copy(), b.copy()),
            n_iter=300,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        std_model.fit(X)
        print(std_model._report_card(**std_model.score(X)))
        print(std_model.reconstruct(X))
        print("Training sequential model...")
        seq_model = SequentialBernoulliRBM(
            params=(a.copy(), W.copy(), b.copy()),
            n_iter=300,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        seq_model.fit(X)
        print(seq_model._report_card(**seq_model.score(X)))
        print(seq_model.reconstruct(X))
    # Done

    # Test transfer learning
    print("Experiment 3: transfer learning")
    for num_bits in range(4, 5):
        print("Training with %d bit%s..." % (num_bits, plural_str(num_bits)))
        X = blib.binary_matrix(range(2 ** num_bits))
        a, W, b = gen_params(num_bits, num_bits)
        print("Training standard RBM...")
        std_model = StandardBernoulliRBM(
            params=(a.copy(), W.copy(), b.copy()),
            n_iter=400,
            n_report=100,
            batch_size=1.0,
            L1_penalty=0.0,
            L2_penalty=0.0,
        )
        std_model.fit(X)
        print(std_model.reconstruct(X))
        print("Testing sequential RBM...")
        seq_model = SequentialBernoulliRBM(params=std_model._params)
        print(seq_model._report_card(**seq_model.score(X)))
        print(seq_model.reconstruct(X))
    # Done
