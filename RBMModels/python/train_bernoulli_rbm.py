"""
Loads the specified data file and trains a Bernoulli RBM.
"""

import sys
import pandas as pd
import numpy as np
from bernoulli_rbm import BernoulliRBM
from bernoulli_lib import bernoulli_decision


if "__main__" == __name__:
    if len(sys.argv) != 3:
        print("Usage: %s <num_hidden> <data_file_path>" % sys.argv[0])
        sys.exit(0)
    num_hidden = int(sys.argv[1])
    data_path = sys.argv[2]
    df = pd.read_csv(data_path, index_col=0)
    num_test = max(int(0.2 * len(df)), 1)
    num_train = len(df) - num_test
    X_train = df.values[0:num_train, :]
    rbm = BernoulliRBM(num_hidden).fit(X_train)
    print("training scores:", rbm._report_card(X_train))
    a, b, W = rbm._params[0:3]
    print("a =", a)
    print("b =", b)
    print("W =", W)
    X_test = df.values[num_train:, :]
    print("testing scores:", rbm._report_card(X_test))
    probs = rbm.output_means(X_test)
    print("testing probabilities:", probs)
    e_train = np.mean(rbm.free_energies(X_train))
    e_test = np.mean(rbm.free_energies(X_test))
    print("Energy: train =", e_train, "test =", e_test, "diff=", e_test - e_train)
    print("training output:")
    print(bernoulli_decision(rbm.output_means(X_train)))
    print("testing output:")
    print(bernoulli_decision(rbm.output_means(X_test)))
