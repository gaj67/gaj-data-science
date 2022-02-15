"""
Tests the various implementations of Restricted Boltzmann Machines.
"""

import numpy as np
import bernoulli_lib as blib
import vector_lib as vlib
from bernoulli_rbm import StandardBernoulliRBM


if "__main__" == __name__:
    # Test case for BRBM: outputs = inputs
    VEC_SIZE = 5
    X = blib.binary_matrix(range(2 ** VEC_SIZE), VEC_SIZE)

    # Hand-build a model designed to replicate the inputs
    a = -0.5 * np.ones(VEC_SIZE)
    b = -0.5 * np.ones(VEC_SIZE)
    W = np.diag(np.ones(VEC_SIZE))
    gen_model = StandardBernoulliRBM(params=(a, W, b), n_iter=25, n_report=1)

    # Test that the generative model reconstructs the inputs
    q = gen_model.predict(X)
    Y = blib.binary_decision(q)
    assert np.all(Y == X)
    print("Hand-crafted model:")
    gen_model.fit(X)

    # Learn random models of the inputs
    print("Independent training runs:")
    for i in range(10):
        lrn_model = StandardBernoulliRBM(VEC_SIZE)
        lrn_model.fit(X)
        report = lrn_model._report_card(**lrn_model.score(X))
        print("Trained random model %d - %s" % (i + 1, report))
