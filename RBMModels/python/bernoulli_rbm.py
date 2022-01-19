"""
Implementations of Restricted Boltzmann Machines with Bernoulli outputs.
"""

import numpy as np

from base_rbm import BaseRBM
import bernoulli_lib as blib
import vector_lib as vlib


class BernoulliRBM(BaseRBM):
    """
    Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary input vector 'x' and
    binary output vector 'y', with energy function:

        E(x,y) = -( a^T x + b^T y + x^T W y ).

    Arguments:
        - n_input (int): The size F of the visible input vector 'x'.
        - n_output (int): The size H of the hidden output vector 'y'.
        - n_iter (int, default=100): The number of passes over the training
            data for parameter estimation.
        - batch_size (int or float, default=0.1): The number or proportion of
            cases with which to compute the gradient per parameter update.
        - step_size (float, default=0.5): The gradient update size.
        - L1_penalty (float, default=0.01): The weight of the L1-norm
            parameter regularisation.
        - L2_penalty (float, default=0.01): The weight of the L2-norm
            parameter regularisation.
        - verbose (bool, default=False): If True, then training information
            will be printed by a call to fit().
        - params (tuple of array, default=None): If specified, provides
            the model parameters, and n_input and n_output are ignored.
            Otherwise, the parameters will be randomly initialised.
    """

    def _set_parameters(self, *params):
        """
        Sets the model parameters from the given values.

        Inputs:
            - a (array): The F-sized vector of visible weights.
            - b (array): The H-sized vector of hidden weights.
            - W (array): The F x H matrix of interaction weights.
        """
        if len(params) < 3:
            raise ValueError("Expected at least 3 parameters")
        self._params = params
        self.n_input = len(params[0])
        self.n_output = len(params[1])

    def _initialise_parameters(self):
        """
        Initialises the model parameters with random values.

        Parameters:
            - a (array): The F-sized vector of visible weights.
            - b (array): The H-sized vector of hidden weights.
            - W (array): The F x H matrix of interaction weights.
        """
        # Initialise the model as E(x,y) = -(x-0.5)^T W (y-0.5)
        W = vlib.random_tensor(self.n_input, self.n_output)
        vlib.orthogonalise_columns(W)
        a = -0.5 * np.sum(W, axis=1)
        b = -0.5 * np.sum(W, axis=0)
        self._params = (a, b, W)

    def _compute_gradients(self, X):
        """
        Computes the approximate gradients of the mean approximate log-likelihood of
        the input data, for each of the model parameters.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - grad_a (array): The size-F vector gradient for 'a'.
            - grad_b (array): The size-H vector gradient for 'b'.
            - grad_W (array): The F x H matrix gradient for 'W'.
        """
        a, b, W = self._params[0:3]
        grad_a, grad_b, grad_W = blib.grad_mean_field(a, b, W, X)
        # Add weight regularisation
        dim = np.prod(W.shape)
        grad_W -= self.L1_penalty * np.sign(W) / dim
        grad_W -= self.L2_penalty * W / dim
        return grad_a, grad_b, grad_W

    def log_probs(self, X):
        """
        Computes the approximate log-likelihoods of the input data.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - scores (array): The size-N array of scores.
        """
        a, b, W = self._params[0:3]
        _, P = blib.reconstruction_probs(a, b, W, X)
        scores = blib.reconstruction_scores(X, P)
        return scores

    def _report_card(self, X, iter=None):
        """
        Computes various mean scores of the input data.

        The root-mean-square error (RMSE) is computed as:

            R = sqrt{ 1/N sum_{d=1}^N sum_{i=1}^F (x_di - p_di)^2 }

        where

            p_di = P(X_i=1|y=[q_dj]), q_dj = P(Y_j=1|x=x_d)

        The mean absolute error (MAE) is computed as:

            E = 1/N sum_{d=1}^N sum_{i=1}^F |x_di - x'_di|

        where

            x'_di = binary_decision(p_di)

        The mean log approximate probability (MLAP) is computed as:

            L = 1/N sum_{d=1}^N log p(x_d)

        Input:
            - X (array): The N x F matrix of input cases.
            - iter (int, optional): The current training iteration number.
        Returns:
            - report (str): The report string.
        """
        a, b, W = self._params[0:3]
        _, P = blib.reconstruction_probs(a, b, W, X)
        score = blib.mean_reconstruction_score(X, P)
        rmse = np.sqrt(blib.mean_reconstruction_error(X, P))
        Xd = vlib.binary_decision(P)
        mae = np.mean(np.sum(np.abs(X - Xd), axis=1))
        template = "score=%f, rmse=%f, mae=%f"
        if iter is not None:
            template = "Iteration " + str(iter) + ": " + template
        return template % (score, rmse, mae)

    def output_means(self, X):
        """
        Computes the output probabilities given the inputs.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - probs (array): The N x H matrix of output probabilities.
        """
        b, W = self._params[1:3]
        return blib.output_probs(b, W, X)

    def input_means(self, Y):
        """
        Computes the input probabilities given the outputs.

        Input:
            - Y (array): The N x H matrix of output cases.
        Returns:
            - probs (array): The N x F matrix of input probabilities.
        """
        a, _, W = self._params[0:3]
        return blib.input_probs(a, W, Y)

    def free_energies(self, X):
        """
        Computes the mean free energy of the dataset.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - F (float): The mean free energy.
        """
        a, b, W = self._params[0:3]
        return blib.free_energies(a, b, W, X)


class SequentialBernoulliRBM(BernoulliRBM):
    """
    Sequential Bernoulli Restricted Boltzmann Machine (RBM).

    Imposes a Markov sequence upon the elements of each input vector, such that

      p(x_1,x_2,...,x_N) = p(x_1) p(x_2|x_1) ... p(x_N|x_1,x_2,...,x_{N-1}).
    """

    def log_probs(self, X):
        """
        Computes the approximate log-likelihoods of the input data, where

          p(x_i|x_1,...,x_{i-1}) = E_{y|x_1,...,x_{i-1}}[ p(x_i|y) ].

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - scores (array): The size-N array of scores.
        """
        a, b, W = self._params[0:3]
        P = blib.sequential_reconstruction_probs(a, b, W, X)
        return blib.reconstruction_scores(X, P)

    def _compute_gradients(self, X):
        a, b, W = self._params[0:3]
        return blib.grad_sequential_reconstruction_probs(a, b, W, X)
        # return blib.approx_grad_sequential_reconstruction_probs(a, b, W, X)

    def _report_card(self, X, iter=None):
        a, b, W = self._params[0:3]
        P = blib.sequential_reconstruction_probs(a, b, W, X)
        score = blib.mean_reconstruction_score(X, P)
        rmse = np.sqrt(blib.mean_reconstruction_error(X, P))
        Xd = vlib.binary_decision(P)
        mae = np.mean(np.sum(np.abs(X - Xd), axis=1))
        template = "score=%f, rmse=%f, mae=%f"
        if iter is not None:
            template = "Iteration " + str(iter) + ": " + template
        return template % (score, rmse, mae)
