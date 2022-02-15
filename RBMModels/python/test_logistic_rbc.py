"""
Tests the various implementations of Restricted Boltzmann Classifiers.
"""

import numpy as np
import bernoulli_lib as blib
import vector_lib as vlib
from bernoulli_rbm import LogisticBernoulliRBC


if "__main__" == __name__:
    # Test basic model
    print("Experiment 1: basic supervised learning")
    X = np.array([(1, 0, 1, 0), (0, 1, 0, 1)])
    Y = np.array([1, 0])
    rbc_model = LogisticBernoulliRBC(
        n_output=2,
        n_iter=25,
        n_report=1,
        batch_size=1.0,
        L1_penalty=0.0,
        L2_penalty=0.0,
    )
    rbc_model.fit(X, Y)
    print(rbc_model.predict(X))
    x_scores = rbc_model.log_probs(X)
    p_x = np.exp(x_scores)
    print("Marginal, p(x)=%s" % str(p_x))
    x_scores, y_scores = rbc_model.log_probs(X, Y)
    p_x = np.exp(x_scores)
    p_y_g_x = np.exp(y_scores)
    p_x_y = p_x * p_y_g_x
    print(
        "Joint, p(x,y)=%s, p(x)=%s, p(y|x)=%s"
        % (str(p_x_y), str(p_x), str(p_y_g_x))
    )
    # Done

    # Test discriminative learning
    print("Experiment 2: discriminative supervised learning")
    rbc_model = LogisticBernoulliRBC(
        is_discriminative=True,
        n_output=2,
        n_iter=25,
        n_report=1,
        batch_size=1.0,
        L1_penalty=0.0,
        L2_penalty=0.0,
    )
    rbc_model.fit(X, Y)
    print(rbc_model.predict(X))
    # Done


    # Test unsupervised learning of the joint log-likelihood,
    # taking note of the expected discriminative scores.
    print("Experiment 3: unsupervised (cluster) learning")
    rbc_model = LogisticBernoulliRBC(
        n_output=2,
        n_iter=25,
        n_report=1,
        batch_size=1.0,
        L1_penalty=0.0,
        L2_penalty=0.0,
    )
    rbc_model.fit(X)
    print(rbc_model.predict(X))
    # Done
