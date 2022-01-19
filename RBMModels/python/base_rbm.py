"""
Base class for implementations of Restricted Boltzmann Machines.
"""

from abc import abstractmethod
import numpy as np


class BaseRBM:
    """
    Restricted Boltzmann Machine (RBM).

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

    def __init__(
        self,
        n_output=None,
        n_input=None,
        *,
        n_iter=100,
        batch_size=0.1,
        step_size=0.5,
        L1_penalty=0.01,
        L2_penalty=0.01,
        verbose=False,
        params=None,
    ):
        if params is not None:
            self._set_parameters(*params)
        else:
            self.n_input = n_input
            self.n_output = n_output
            if n_input is not None:
                self._initialise_parameters()
            # else wait for fit()
        self.n_iter = n_iter
        if batch_size <= 0 or isinstance(batch_size, float) and batch_size > 1:
            raise ValueError(f"Invalid batch_size: {batch_size}")
        self.batch_size = batch_size
        self.step_size = step_size
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty
        self.verbose = verbose

    @abstractmethod
    def _set_parameters(self, *params):
        """
        Sets the model parameters, self._params, from the given values.
        NOTE: Must also set self.n_input and self.n_output.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialise_parameters(self):
        """
        Initialises the model parameters, self._params.
        NOTE: Assumes self.n_input and self.n_output are already set.
        """
        raise NotImplementedError

    def fit(self, X, y=None):
        """
        Estimates the model parameters from the training data.

        Inputs:
            - X (array): The N x F matrix of input cases.
            - y (array): The size-N vector of input labels. This is ignored
                for unsupervised learning.
        """
        if self.n_input is None:
            self.n_input = X.shape[1]
            self._initialise_parameters()
        if self.verbose:
            print(self._report_card(X, 0))
        n_samples = X.shape[0]
        batch_size = self.batch_size
        if isinstance(batch_size, float):
            batch_size = max(int(batch_size * n_samples), 1)
        n_batches = (n_samples + batch_size - 1) // batch_size
        for i in range(1, self.n_iter + 1):
            batch_start = 0
            for j in range(n_batches):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_data = X[batch_start:batch_end, :]
                self._update_parameters(batch_data)
                batch_start += batch_size
            if self.verbose:
                print(self._report_card(X, i))
        return self

    def _update_parameters(self, X):
        """
        Updates the model parameters given the input data.

        Inputs:
            - X (array): The N x F matrix of input cases.
        """
        # Compute new direction as a function of the gradient
        grads = self._compute_gradients(X)
        rho = self.step_size
        for i, param in enumerate(self._params):
            param += rho * grads[i]

    @abstractmethod
    def _compute_gradients(self, X):
        """
        Computes the approximate gradient of the mean log-likelihood
        for each of the model parameters.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - grads (tuple of array): The gradients.
        """
        raise NotImplementedError

    def _report_card(self, X, iter=None):
        """
        Computes information about the current likelihood of the input data.

        Inputs:
            - X (array): The N x F matrix of input cases.
            - iter (int, optional): The current training iteration number.
        Returns:
            - report (str): The report string.
        """
        score = np.mean(self.log_probs(X))
        if iter is not None:
            return "Iteration %d: score=%f" % (iter, score)
        else:
            return "score=%f" % score

    @abstractmethod
    def log_probs(self, X):
        """
        Computes the approximate log-probability of each input vector.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - scores (array): The size-N array of scores.
        """
        raise NotImplementedError

    @abstractmethod
    def output_means(self, X):
        """
        Computes the expected values of the outputs given the inputs.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - probs (array): The N x H matrix of output means.
        """
        raise NotImplementedError

    @abstractmethod
    def input_means(self, Y):
        """
        Computes the expected values of the inputs given the outputs.

        Input:
            - Y (array): The N x H matrix of output cases.
        Returns:
            - probs (array): The N x F matrix of input probabilities.
        """
        raise NotImplementedError

    @abstractmethod
    def free_energies(self, X):
        """
        Computes the free energy of each input vector.

        Input:
            - X (array): The N x F matrix of input cases.
        Returns:
            - F (array): The N-size vector of free energies.
        """
        raise NotImplementedError
