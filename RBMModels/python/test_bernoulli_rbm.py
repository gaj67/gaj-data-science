"""
Tests the various implementations of Restricted Boltzmann Machines.
"""

import numpy as np
import bernoulli_lib as blib
import vector_lib as vlib
from bernoulli_rbm import BernoulliRBM


if "__main__" == __name__:
    # Test case for BRBM: outputs = inputs
    VEC_SIZE = 5
    X = vlib.binary_matrix(range(2 ** VEC_SIZE), VEC_SIZE)

    # Hand-build a model designed to replicate the inputs
    a = -0.5 * np.ones(VEC_SIZE)
    b = -0.5 * np.ones(VEC_SIZE)
    W = np.diag(np.ones(VEC_SIZE))
    gen_model = BernoulliRBM(params=(a, b, W), verbose=True, n_iter=25)

    # Test that the generative model reconstructs the inputs
    q = gen_model.output_means(X)
    Y = vlib.binary_decision(q)
    assert np.all(Y == X)
    print("Hand-crafted model:")
    gen_model.fit(X)

    # Learn random models of the inputs
    print("Independent training runs:")
    for i in range(10):
        lrn_model = BernoulliRBM(VEC_SIZE).fit(X)
        print("Trained random model %d - %s" % (i + 1, lrn_model._report_card(X)))
