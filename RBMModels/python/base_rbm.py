"""
Base class for implementations of Restricted Boltzmann Machines.
"""

from abc import abstractmethod
import numpy as np
import vector_lib as vlib


class RBM:
    """
    Base class for Restricted Boltzmann Machines (RBMs) and Restricted
    Boltzmann Classifiers (RBCs).

    An RBM is a Boltzmann Machine with a sequence of layers, such that
    each consecutive pair of layers is connected as a bipartite graph.
    This restriction imposes useful conditional independence between variables.

    The primary purpose of an RBM is as a predictive model, where the
    'forward' or 'predictive' computations proceed from vector input 'x'
    to vector output 'y', via zero, one or more hidden layers.

    If the output takes the form of a (binary) one-hot vector, then the RBM is
    also a classifier (or RBC, for short).
    """

    def __init__(
        self,
        params=None,
        is_RBC=False,
        is_discriminative=False,
        n_iter=100,
        n_report=0,
        batch_size=0.1,
        step_size=0.5,
        step_decay=0.0,
        L1_penalty=0.0,
        L2_penalty=0.0,
    ):
        """
        Inputs:
            - params (RBM or tuple of array, optional): If specified, then the
                parameter dimensions are set from the given parameters.
            - is_RBC (bool, default=False): Indicates whether (True) or not
                (False) the RBM is also a classifier (RBC).
            - is_discriminative (bool, default=False): If True, indicates that
                supervised training should optimise the discriminative
                likelihood p(y|x); otherwise, the joint likelihood p(x,y)
                will be optimised.
            - n_iter (int, default=100): The number of passes over the training
                data for parameter estimation.
            - n_report (int, default=0): If positive, then training information
                will be printed by calls to fit() every 'n_report' major
                iterations.
            - batch_size (int or float, default=0.1): The number or proportion of
                cases with which to compute the gradient per parameter update.
            - step_size (float, default=0.5): The gradient update size,
                or learning rate.
            - step_decay (float, default=0.0): The 'step_size' is multiplied
                by (1 - 'step_decay') after each training iteration, to help
                prevent gradient over-shooting. Typically, one would choose
                a conservative value like 10^{-k}, with k >= 3.
            - L1_penalty (float, default=0.0): The weight of the L1-norm
                parameter regularisation.
            - L2_penalty (float, default=0.0): The weight of the L2-norm
                parameter regularisation.
        """
        if params is not None:
            if isinstance(params, RBM):
                params = params.get_parameters()
            self._set_parameters(*params)
        self._is_RBC = is_RBC
        self._is_discriminative = is_discriminative
        # Training options:
        self.n_iter = n_iter
        self.n_report = n_report
        self.batch_size = batch_size
        self.step_size = step_size
        assert 0.0 <= step_decay < 1.0
        self.step_decay = step_decay
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty

    def is_RBC(self):
        """
        Indicates whether or not the RBM is also a classifier (RBC).

        Returns:
            - flag (bool): A value of True (else False) if the model is
                an RBC.
        """
        return self._is_RBC

    def is_discriminative(self):
        """
        Indicates whether supervised training should maximise the
        discriminative likelihood, p(y|x), or the joint likelihood, p(x,y).

        Returns:
            - flag (bool): A value of True (else False) if supervised training
            should be discriminative.
        """
        return self._is_discriminative

    def fit(self, X, Y=None):
        """
        Estimates the model parameters from the training data.

        Inputs:
            - X (array): The matrix of row-wise input cases.
            - Y (array, optional): A vector of output labels (RBC), or a matrix
                of row-wise output cases (RBM), if supervised learning is
                permitted.
        """
        if self.n_report > 0:
            print(self._report_card(iter=0, **self.score(X, Y)))
        n_samples = X.shape[0]
        batch_size = self.batch_size
        if isinstance(batch_size, float):
            batch_size = max(int(batch_size * n_samples), 1)
        n_batches = (n_samples + batch_size - 1) // batch_size
        if Y is None:
            _get_y_batch = lambda s, e: None
        elif vlib.is_matrix(Y):
            _get_y_batch = lambda s, e: Y[s:e, :]
        else:
            _get_y_batch = lambda s, e: Y[s:e]
        for i in range(1, self.n_iter + 1):
            batch_start = 0
            for j in range(n_batches):
                batch_end = min(batch_start + batch_size, n_samples)
                X_batch = X[batch_start:batch_end, :]
                y_batch = _get_y_batch(batch_start, batch_end)
                self._update_parameters(X_batch, y_batch)
                batch_start += batch_size
            if self.n_report > 0 and i % self.n_report == 0:
                print(self._report_card(iter=i, **self.score(X, Y)))
            self.step_size *= (1 - self.step_decay)
        if self.n_report > 0 and i % self.n_report > 0:
            print(self._report_card(iter=i, **self.score(X, Y)))

    @abstractmethod
    def predict(self, X):
        """
        Computes the expected values of the outputs given the inputs.

        Input:
            - X (array): The matrix of row-wise input cases.
        Returns:
            - Y_bar (array): The matrix of row-wise expected outputs.
        """
        # NOTE: It will be more efficient to skip the reconstruction step.
        X_bar, Y_bar = self.predict_reconstruct(X)
        return Y_bar

    @abstractmethod
    def reconstruct(self, X):
        """
        Computes an approximation of the expected input values
        given the observed values.

        Input:
            - X (array): The matrix of row-wise input cases.
        Returns:
            - X_bar (array): The matrix of row-wise reconstructed inputs.
        """
        X_bar, Y_bar = self.predict_reconstruct(X)
        return X_bar

    @abstractmethod
    def predict_reconstruct(self, X):
        """
        Computes an approximation of the expected input and output values
        given the observed values.

        Input:
            - X (array): The matrix of row-wise input cases.
        Returns:
            - X_bar (array): The matrix of row-wise reconstructed inputs.
            - Y_bar (array): The matrix of row-wise expected outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def log_probs(self, X, Y=None):
        """
        Computes the approximate log-likelihoods of each data case.
        If y is specified, then log p(y|x) is computed along with log p(x),
        otherwise just log p(x) is computed.

        Input:
            - X (array): The matrix of row-wise input cases.
            - Y (array, optional): A vector of output labels (RBC), or a matrix
                of row-wise output cases (RBM), if supervised learning is
                permitted.
        Returns:
            - x_scores (array): The vector of log-likelihoods, log p(x).
            - y_scores (array): The vector of log-likelihoods, log p(y|x)
                (not returned if Y is None).
        """
        raise NotImplementedError

    def score(self, X, Y=None):
        """
        Computes scoring information about the likelihood of the
        specified data given the current model parameter values.

        The scoring information includes at least:
            - 'x_score': The mean marginal log-likelihood, log p(x),
                of the specified input data.
            - 'rmse': The root mean square error of the reconstucted input
                data compared to the actual input data.
        In addition, if output Y is specified, then the scoring information
        also contains:
            - 'xy_score': The mean joint log-likelihood, log p(x,y),
                of the input data.
            - 'y_score': The mean discriminative log-likelihood, log p(y|x),
                of the output data conditioned on the input data.

        Inputs:
            - X (array): The matrix of row-wise input cases.
            - Y (array, optional): A vector of output labels (RBC), or a matrix
                of row-wise output cases (RBM), if supervised learning is
                permitted.
        Returns:
            - scores (dict): The dictionary of scoring information.
        """
        scores = {}
        if Y is None:
            # Unsupervised
            scores['x_score'] = np.mean(self.log_probs(X))
        else:
            # Supervised
            x_scores, y_scores = self.log_probs(X, Y)
            scores['x_score'] = x_score = np.mean(x_scores)
            scores['y_score'] = y_score = np.mean(y_scores)
            scores['xy_score'] = x_score + y_score
        X_bar = self.reconstruct(X)
        scores['rmse'] = np.sqrt(np.mean(vlib.square_errors(X, X_bar)))
        return scores

    @abstractmethod
    def _set_parameters(self, *params):
        """
        Sets the model parameters and the model dimensions
        from the given values.
        """
        raise NotImplementedError

    @abstractmethod
    def _is_dimensioned(self):
        """
        Checks if the model dimensions are set.

        Returns:
            - flag (bool): A value of True (else False) if all parameter
                dimensions are known.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialise_parameters(self):
        """
        Initialises the model parameters.
        NOTE: Assumes all parameter dimensions are already set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        """
        Obtains the model parameters.

        Returns:
            - params: A tuple of the model parameters.
        """
        raise NotImplementedError

    def _update_parameters(self, X, Y=None):
        """
        Updates the model parameters given the input data.

        Inputs:
            - X (array): The matrix of row-wise input cases.
            - Y (array, optional): A vector of output labels (RBC), or a matrix
                of row-wise output cases (RBM), if supervised learning is
                permitted.
        """
        # Compute new direction as a function of the gradient
        grads = self._compute_score_gradients(X, Y)
        # Allow for L1 parameter penalisation
        if self.L1_penalty > 0:
            penalties = self._compute_L1_gradients()
            for grad, penalty in zip(grads, penalties):
                if penalty is None or grad is None:
                    continue
                # Prevent the penalty from overwhelming the gradient
                ind = (penalty != 0) & (np.sign(penalty) == np.sign(grad))
                if np.any(ind):
                    m = np.min(grad[ind] / penalty[ind])
                    grad -= self.L1_penalty * m * penalty
        # Allow for L2 parameter penalisation
        if self.L2_penalty > 0:
            penalties = self._compute_L2_gradients()
            for grad, penalty in zip(grads, penalties):
                if penalty is None or grad is None:
                    continue
                # Prevent the penalty from overwhelming the gradient
                ind = (penalty != 0) & (np.sign(penalty) == np.sign(grad))
                if np.any(ind):
                    m = np.min(grad[ind] / penalty[ind])
                    grad -= self.L2_penalty * m * penalty
        # Step in the direction of the gradient
        rho = self.step_size
        for param, grad in zip(self.get_parameters(), grads):
            if grad is not None:
                param += rho * grad

    @abstractmethod
    def _compute_score_gradients(self, X, Y=None):
        """
        Computes the approximate gradient of the mean log-likelihood
        for each of the model parameters.

        Input:
            - X (array): The matrix of input row vectors.
            - Y (array, optional): A vector of output labels (RBC), or a matrix
                of output row vectors (RBM), if supervised learning is
                permitted.
        Returns:
            - grads (tuple of array): The gradients.
        """
        raise NotImplementedError

    def _compute_L1_gradients(self):
        """
        Computes the gradient of the L1 penalty term
        for each of the model parameters.

        By default, for parameter theta the L1 penalty is |theta|, with gradient
        sign(theta). However, it is common for bias parameters to not be
        penalised - in such a case, None may be returned in place of a gradient.

        Returns:
            - grads (tuple of array): The gradients.
        """
        return tuple(np.sign(p) for p in self.get_parameters())

    def _compute_L2_gradients(self):
        """
        Computes the gradient of the L2 penalty term
        for each of the model parameters.

        By default, for parameter theta the L2 penalty is 0.5 ||theta||^2,
        with gradient theta. However, it is common for bias parameters to not be
        penalised - in such a case, None may be returned in place of a gradient.

        Returns:
            - grads (tuple of array): The gradients.
        """
        return self.get_parameters()

    def _report_card(self, iter=None, **scores):
        """
        Collates scoring information about the likelihood of the
        specified data.

        Inputs:
            - iter (int, optional): The current training iteration number.
            - scores (dict): The dictionary of scoring information.
        Returns:
            - report (str): The report format string.
        """
        report = "x_score={x_score:f}, rmse={rmse:f}"
        if 'y_score' in scores:
            # Supervised
            report = "xy_score={xy_score:f}, y_score={y_score:f}, " + report
        if iter is not None:
            report = "Iteration " + str(iter) + ": " + report
        return report.format(**scores)


###############################################################################

class RBM2(RBM):
    """
    A two-layer RBM with input 'x' and output 'y'.
    The simplest such model has energy function

            E(x,y) = -( a^T x + x^T W y + b^T y ),

    although in practice the 'x' on the right-hand side could be replaced
    by a nonlinear, deterministic function of the actual input 'x' on the
    left-hand side.
    """

    def __init__(self, n_output=None, n_input=None, **kwds):
        """
        Arguments:
            - n_output (int, optional): The size of the output vector.
                This need not be supplied if 'params' is given, or
                if supervised training via fit() will subsequently be used.
            - n_input (int, optional): The size of the input vector.
                This need not be supplied if 'params' is given, or
                if supervised or unsupervised training via fit() will
                subsequently be used.
        """
        RBM.__init__(self, **kwds)
        if getattr(self, "n_output", None) is None:
            self.n_output = n_output
        if getattr(self, "n_input", None) is None:
            self.n_input = n_input
        if getattr(self, "_params", None) is None and self._is_dimensioned():
            self._initialise_parameters()

    def _is_dimensioned(self):
        return (
            getattr(self, "n_output", None) is not None
            and getattr(self, "n_input", None) is not None
        )

    def _set_parameters(self, *params):
        """
        Sets the model parameters from the given values.

        Inputs:
            - a (array): The vector of input biases.
            - W (array): The matrix of input/output weights.
            - b (array): The vector of output biases.
        """
        if len(params) != 3:
            raise ValueError("Expected 3 parameters")
        self._params = params
        a, W, b = params
        self.n_input = len(a)
        self.n_output = len(b)

    def _initialise_parameters(self):
        """
        Initialises the model parameters with random values.

        Parameters:
            - a (array): The vector of input biases.
            - W (array): The matrix of input/output weights.
            - b (array): The vector of output biases.
        """
        # Initialise the model as E(x,y) = -x^T W z
        W = vlib.random_tensor(self.n_input, self.n_output)
        vlib.orthogonalise_columns(W)
        a = np.zeros(self.n_input)
        b = np.zeros(self.n_output)
        self._params = (a, W, b)

    def get_parameters(self):
        """
        Obtains the model parameters.

        Returns:
            - a (array): The vector of input biases.
            - W (array): The matrix of input/output weights.
            - b (array): The vector of output biases.
        """
        return self._params

    def _compute_L1_gradients(self):
        # Do not penalise the bias parameters
        a, W, b = self._params
        return None, np.sign(W), None

    def _compute_L2_gradients(self):
        # Do not penalise the bias parameters
        a, W, b = self._params
        return None, W, None

    def fit(self, X, Y=None):
        # Attempt to initialise deferred parameters from data
        if getattr(self, "n_input", None) is None:
            self.n_input = X.shape[1]
        if getattr(self, "n_output", None) is None:
            if Y is None:
                raise ValueError("Unknown number of outputs")
            elif vlib.is_matrix(Y):
                self.n_output = Y.shape[1]
            elif self.is_RBC():
                # XXX Assumes all class labels are present!
                self.n_output = 1 + max(Y)
            else:
                raise ValueError("Unknown number of outputs")
        if not self._is_dimensioned():
            raise ValueError("Cannot determine parameter dimensions")
        if getattr(self, "_params", None) is None:
            self._initialise_parameters()
        # Now perform learning
        super().fit(X, Y)


###############################################################################

class RBM3(RBM):
    """
    A three-layer RBM with input 'x', hidden layer 'z', and output 'y'.
    The simplest such model has energy function

        E(x,y,z) = -( a^T x + x^T W z + b^T z + z^T U y + c^T y ).

    although in practice the 'x' on the right-hand side could be replaced
    by a nonlinear, deterministic function of the actual input 'x' on the
    left-hand side.
    """

    def __init__(self, n_hidden=None, n_output=None, n_input=None, **kwds):
        """
        Arguments:
            - n_hidden (int, optional): The size of the hidden layer.
                This need not be supplied if 'params' is given.
            - n_output (int, optional): The size of the output vector.
                This need not be supplied if 'params' is given, or
                if supervised training via fit() will subsequently be used.
            - n_input (int, optional): The size of the input vector.
                This need not be supplied if 'params' is given, or
                if supervised or unsupervised training via fit() will
                subsequently be used.
        """
        RBM.__init__(self, **kwds)
        if getattr(self, "n_hidden", None) is None:
            self.n_hidden = n_hidden
        if getattr(self, "n_output", None) is None:
            self.n_output = n_output
        if getattr(self, "n_input", None) is None:
            self.n_input = n_input
        if self.n_hidden is None:
            raise ValueError("Unknown number of hidden units")
        if getattr(self, "_params", None) is None and self._is_dimensioned():
            self._initialise_parameters()

    def _is_dimensioned(self):
        return (
            getattr(self, "n_output", None) is not None
            and getattr(self, "n_input", None) is not None
            and getattr(self, "n_hidden", None) is not None
        )

    def _set_parameters(self, *params):
        """
        Sets the model parameters from the given values.

        Inputs:
            - a (array): The vector of input biases.
            - W (array): The matrix of input/hidden weights.
            - b (array): The vector of hidden biases.
            - U (array): The matrix of hidden/output weights.
            - c (array): The vector of output biases.
        """
        if len(params) != 5:
            raise ValueError("Expected 5 parameters")
        self._params = params
        a, W, b, U, c = params
        self.n_input = len(a)
        self.n_hidden = len(b)
        self.n_output = len(c)

    def _initialise_parameters(self):
        """
        Initialises the model parameters with random values.

        Parameters:
            - a (array): The vector of input biases.
            - W (array): The matrix of input/hidden weights.
            - b (array): The vector of hidden biases.
            - U (array): The matrix of hidden/output weights.
            - c (array): The vector of output biases.
        """
        # Initialise the model as E(x,y) = -x^T W z - z^T U y
        W = vlib.random_tensor(self.n_input, self.n_hidden)
        vlib.orthogonalise_columns(W)
        a = np.zeros(self.n_input)
        b = np.zeros(self.n_hidden)
        U = vlib.random_tensor(self.n_hidden, self.n_output)
        vlib.orthogonalise_columns(U)
        c = np.zeros(self.n_output)
        self._params = (a, W, b, U, c)

    def get_parameters(self):
        """
        Obtains the model parameters.

        Returns:
            - a (array): The vector of input biases.
            - W (array): The matrix of input/hidden weights.
            - b (array): The vector of hidden biases.
            - U (array): The matrix of hidden/output weights.
            - c (array): The vector of output biases.
        """
        return self._params

    def _compute_L1_gradients(self):
        # Do not penalise the bias parameters
        a, W, b, U, c = self._params
        return None, np.sign(W), None, np.sign(U), None

    def _compute_L2_gradients(self):
        # Do not penalise the bias parameters
        a, W, b, U, c = self._params
        return None, W, None, U, None

    def fit(self, X, Y=None):
        # Attempt to initialise deferred parameters from data
        if getattr(self, "n_input", None) is None:
            self.n_input = X.shape[1]
        if getattr(self, "n_output", None) is None:
            if Y is None:
                raise ValueError("Unknown number of outputs")
            elif vlib.is_matrix(Y):
                self.n_output = Y.shape[1]
            elif self.is_RBC():
                # XXX Assumes all class labels are present!
                self.n_output = 1 + max(Y)
            else:
                raise ValueError("Unknown number of outputs")
        if not self._is_dimensioned():
            raise ValueError("Cannot determine parameter dimensions")
        if getattr(self, "_params", None) is None:
            self._initialise_parameters()
        # Now perform learning
        super().fit(X, Y)
