"""
Implementations of Restricted Boltzmann Machines with Bernoulli outputs.
"""

import numpy as np

from base_rbm import RBM, RBM2, RBM3
import bernoulli_lib as blib
import vector_lib as vlib


class BernoulliRBM(RBM):
    """
    Base class for Bernoulli Restricted Boltzmann Machines (RBMs) and
    Bernoullic Restricted Boltzmann Classifiers (RBCs).

    The primary assumption is that all layers take binary valued vectors.

    The secondary assumption is that, for input 'x' and output 'y', the score
    being maximised is the joint log-likelihood, log p(x,y), for supervised
    training, or the marginal log-likelihood, log p(x), for unsupervised
    training.

    If discriminative training is required, i.e. we wish to maximise the
    discriminative log-likelihood, log p(y|x), then be aware that some parameters
    might not be able to be estimated (particularly any bias term for 'x').
    """

    def log_probs(self, X, Y=None):
        if Y is None:
            # Unsupervised
            X_bar = self.reconstruct(X)
            x_scores = blib.binary_scores(X, X_bar)
            return x_scores
        else:
            # Supervised
            X_bar, Y_bar = self.predict_reconstruct(X)
            x_scores = blib.binary_scores(X, X_bar)
            if self.is_RBC():
                y_scores = blib.one_hot_scores(Y, Y_bar)
            else:
                y_scores = blib.binary_scores(Y, Y_bar)
            return x_scores, y_scores

    def score(self, X, Y=None):
        scores = {}
        X_bar, Y_bar = self.predict_reconstruct(X)
        scores['x_score'] = x_score = np.mean(blib.binary_scores(X, X_bar))
        scores['rmse'] = np.sqrt(np.mean(vlib.square_errors(X, X_bar)))
        scores['mae'] = np.mean(blib.binary_errors(X, X_bar))
        if Y is None:
            # Unsupervised - force expected values of log p(y|x)
            Y = Y_bar
        if self.is_RBC():
            y_score = np.mean(blib.one_hot_scores(Y, Y_bar))
        else:
            y_score = np.mean(blib.binary_scores(Y, Y_bar))
        scores['y_score'] = y_score
        scores['xy_score'] = x_score + y_score
        return scores

    def _report_card(self, iter=None, **scores):
        report = super()._report_card(iter, **scores)
        report += ", mae=%f" % scores['mae']
        return report


###############################################################################
class BernoulliRBM2(BernoulliRBM, RBM2):
    """
    Two-layer Bernoulli Restricted Boltzmann Machine.

    The RBM has a binary input vector 'x' and
    binary output vector 'y', with energy function:

        E(x,y) = -( a^T x + x^T W y + b^T y ).
    """

    #def __init__(self, n_output=None, n_input=None, **kwds):
    #    RBM2.__init__(self, n_output, n_input, **kwds)

    def predict(self, X):
        a, W, b = self.get_parameters()
        if self.is_RBC():
            Y_bar = blib.forward_softmax(X, W, b)
        else:
            Y_bar = blib.forward_logistic(X, W, b)
        # bar{y} = E_{y|x}[y]
        return Y_bar

    def predict_reconstruct(self, X):
        # bar{y} = E_{y|x}[y]
        Y_bar = self.predict(X)
        a, W, b = self.get_parameters()
        # bar{x}' = E_{y|x}[ E_{x'|y}[x'] ]
        X_bar = blib.backward_logistic(Y_bar, a, W)
        return X_bar, Y_bar

    def _compute_score_gradients(self, X, Y=None):
        # Compute data terms and reconstructed terms
        a, W, b = self.get_parameters()
        if Y is None:
            # Unsupervised
            # bar{y} = E_{y|x}[y]
            Y = self.predict(X)
            # bar{x}' = E_{y|x}[ E_{x'|y}[x'] ]
            X_c = blib.backward_logistic(Y, a, W)
            # bar{y}' = E_{y|x}[ E_{x'|y}[ E_{y'|x'}[y'] ] ]
            Y_c = self.predict(X_c)
        else:
            # Supervised
            # bar{y}' = E_{y'|x}[y']
            Y_c = self.predict(X)
            # Technically, for discriminative training there is no X_c (or,
            # equivalently, X_c = X), and so grad_a = 0. However, we shall allow
            # a hybrid update grad_a to improve p(x), since the bias parameter
            # 'a' does not affect p(y|x).
            # bar{x}' = E_{y'|x}[ E_{x'|y'}[x'] ]
            X_c = blib.backward_logistic(Y_c, a, W)
        # Compute differences of expectations
        grad_a = np.mean(X, axis=0) - np.mean(X_c, axis=0)
        grad_b = np.mean(Y, axis=0) - np.mean(Y_c, axis=0)
        if self.is_discriminative():
            # Enforce discriminative estimation of 'b' and 'W'.
            X_c = X
        N = X.shape[0]
        if self.is_RBC():
            C = len(b)
            grad_W = (
                vlib.multiply_one_hot(X.T, Y, C) - np.matmul(X_c.T, Y_c)
            ) / N
        else:
            grad_W = (np.matmul(X.T, Y) - np.matmul(X_c.T, Y_c)) / N
        # Provide gradients
        return grad_a, grad_W, grad_b


###############################################################################
class StandardBernoulliRBM(BernoulliRBM2):
    """
    Two-layer Standard Bernoulli Restricted Boltzmann Machine.

    The standard RBM has a binary input vector 'x' and
    binary output vector 'y', with energy function:

        E(x,y) = -( a^T x + x^T W y + b^T y ).
    """
    pass


###############################################################################
class SequentialBernoulliRBM(BernoulliRBM2):
    """
    Two-layer Sequential Bernoulli Restricted Boltzmann Machine.

    The standard RBM has a binary input vector 'x' and
    binary output vector 'y', with energy function:

        E(x,y) = -( a^T x + x^T W y + b^T y ).

    However, a Markov sequence is imposed upon the elements of the input
    vector 'x', such that

      p(x_1,x_2,...,x_F) = p(x_1) p(x_2|x_1) ... p(x_F|x_1,x_2,...,x_{F-1}).
    """

    def reconstruct(self, X):
        a, W, b = self.get_parameters()
        N, F = X.shape
        X_bar = np.zeros((N, F))
        b_plus_XW = np.tile(b, (N, 1))  # N x H array of partial sums
        _fwd = (
            blib._forward_softmax if self.is_RBC()
            else blib._forward_logistic
        )
        _bwd = blib.backward_logistic
        for i in range(F):
            Y_bar = _fwd(b_plus_XW)
            X_bar[:, i : i + 1] = _bwd(Y_bar, a[i : i + 1], W[i : i + 1, :])
            b_plus_XW += np.outer(X[:, i], W[i, :])
        return X_bar

    def predict_reconstruct(self, X):
        X_bar = self.reconstruct(X)
        Y_bar = self.predict(X)
        return X_bar, Y_bar

    def _compute_score_gradients(self, X, Y=None):
        # Assume unsupervised - ignore Y
        a, W, b = self.get_parameters()
        N, F = X.shape
        H = len(b)
        grad_a = np.zeros(F)
        grad_W = np.zeros((F, H))
        sum_B = np.zeros((N, H))  # N x H array of partial sums
        b_plus_XW = np.matmul(X, W) + b  # N x H array of partial sums
        _fwd = (
            blib._forward_softmax if self.is_RBC()
            else blib._forward_logistic
        )
        _bwd = blib.backward_logistic
        for i in range(F - 1, -1, -1):
            x_i = X[:, i]
            w_i = W[i, :]
            b_plus_XW -= np.outer(x_i, w_i)
            Y_bar = _fwd(b_plus_XW)
            x_bar_i = _bwd(Y_bar, a[i : i + 1], W[i : i + 1, :])[:, 0]
            d_i = x_i - x_bar_i
            grad_a[i] = np.mean(d_i)
            grad_W[i, :] = (np.matmul(d_i, Y_bar) + np.matmul(x_i, sum_B)) / N
            B = vlib.multiply_columns(Y_bar * (1 - Y_bar) * w_i, d_i)
            sum_B += B
        grad_b = np.mean(sum_B, axis=0)
        # Provide gradients
        return grad_a, grad_W, grad_b


###############################################################################
class LogisticBernoulliRBC(BernoulliRBM2):
    """
    Two-layer (Logistic) Bernoulli Restricted Boltzmann Classifier.

    A logistic RBC with binary input vector 'x', and (binary) one-hot output
    vector 'y', with energy function:

        E(x,y,z) = -( a^T x + x^T W y + b^T y ).
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, is_RBC=True, **kwds)


###############################################################################
class SequentialBernoulliRBC(SequentialBernoulliRBM):
    """
    Two-layer (Logistic) Sequential Bernoulli Restricted Boltzmann Classifier.

    A logistic RBC with binary input vector 'x', and (binary) one-hot output
    vector 'y', with energy function:

        E(x,y,z) = -( a^T x + x^T W y + b^T y ).

    Furthermore, a Markov sequence is imposed upon the elements of the input
    vector 'x', such that

      p(x_1,x_2,...,x_F) = p(x_1) p(x_2|x_1) ... p(x_F|x_1,x_2,...,x_{F-1}).
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, is_RBC=True, **kwds)


###############################################################################
class BernoulliRBM3(BernoulliRBM, RBM3):
    """
    Three-layer Bernoulli Restricted Boltzmann Machine.

    An RBM with binary input vector 'x', hidden binary
    vector 'z', and binary output vector 'y', with energy function:

        E(x,y,z) = -( a^T x + x^T W z + b^T z + z^T U y + c^T y ).
    """

    #def __init__(self, n_hidden=None, n_output=None, n_input=None, **kwds):
    #    RBM3.__init__(self, n_hidden, n_output, n_input, **kwds)

    def predict(self, X):
        a, W, b, U, c = self.get_parameters()
        if self.is_RBC():
            Y_bar = blib.hidden_softmax(X, W, b, U, c)
        else:
            Y_bar = blib.hidden_logistic(X, W, b, U, c)
        # bar{y} = E_{y|x}[y]
        return Y_bar

    def predict_reconstruct(self, X):
        # bar{y} = E_{y|x}[y]
        Y_bar = self.predict(X)
        a, W, b, U, c = self.get_parameters()
        # bar{z} = E_{y|x}[ E_{z|x,y}[z] ]
        Z_bar = blib.bi_logistic(X, Y_bar, W, b, U)
        # bar{x}' = E_{y|x}[ E_{z|x,y}[ E_{x'|z}[x'] ] ]
        X_bar = blib.backward_logistic(Z_bar, a, W)
        return X_bar, Y_bar

    def _compute_score_gradients(self, X, Y=None):
        # Compute data terms and reconstructed terms
        a, W, b, U, c = self.get_parameters()
        if Y is None:
            # Unsupervised - optimise the marginal log-likelihood
            # bar{y} = E_{y|x}[y]
            Y_d = self.predict(X)
        else:
            # Supervised
            Y_d = Y
        # bar{z} = E_{y|x}[ E_{z|x,y}[z] ]
        Z_d = blib.bi_logistic(X, Y_d, W, b, U)
        # bar{x}' = E_{y|x}[ E_{z|x,y}[ E_{x'|z}[x'] ] ]
        X_c = blib.backward_logistic(Z_d, a, W)
        grad_a = np.mean(X, axis=0) - np.mean(X_c, axis=0)
        if Y is None:
            # bar{y}' = E_{y|x}[ E_{z|x,y}[ E_{x'|z}[ E_{y'|x'}[y'] ] ] ]
            Y_c = self.predict(X_c)
        elif self.is_discriminative():
            # There is no X_c - grad_a is a hybrid approx.
            X_c = X
            # bar{y}' = E_{y'|x}[y']
            Y_c = self.predict(X)
        elif self.is_RBC():
            # bar{y}' = E_{z|x,y}[ E_{y'|z}[y'] ]
            Y_c = blib.forward_softmax(Z_d, U, c)
        else:
            # bar{y}' = E_{z|x,y}[ E_{y'|z}[y'] ]
            Y_c = blib.forward_logistic(Z_d, U, c)
        # bar{z}' = E_{z|x,y}[ E_{x',y'|z}[ E_{z'|x',y'}[z'] ] ]
        Z_c = blib.bi_logistic(X_c, Y_c, W, b, U)
        # Compute differences of expectations
        N = X.shape[0]
        C = len(c)
        grad_b = np.mean(Z_d, axis=0) - np.mean(Z_c, axis=0)
        grad_W = (np.matmul(X.T, Z_d) - np.matmul(X_c.T, Z_c)) / N
        grad_c = vlib.one_hot_means(Y_d, C) - np.mean(Y_c, axis=0)
        grad_U = (
            vlib.multiply_one_hot(Z_d.T, Y_d, C) - np.matmul(Z_c.T, Y_c)
        ) / N
        # Provide gradients
        return grad_a, grad_W, grad_b, grad_U, grad_c


###############################################################################
class StandardBernoulliRBC(BernoulliRBM3):
    """
    Three-layer Bernoulli Restricted Boltzmann Classifier.

    An RBC with binary input vector 'x', hidden binary
    vector 'z', and (binary) one-hot output vector 'y', with energy function:

        E(x,y,z) = -( a^T x + x^T W z + b^T z + z^T U y + c^T y ).
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, is_RBC=True, **kwds)
