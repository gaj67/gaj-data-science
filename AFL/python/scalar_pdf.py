"""
This module defines the base class for a probability distribution function (PDF)
of a (discrete or continuous) scalar variate, in terms of one or more
distributional parameters.

Each parameter may independently be specified as either a scalar value or
a numpy array of values (see the Value type). This also holds true for
the value(s) of the response varriate.

Note that when we specify an input type of 'float', we really mean 'float-like',
i.e. any scalar value that is compatible with float, inluding 'int' and the
various numpy types.

For parameter estimation from observed data, we assume the log-likelihood is
maximised, and thus require derivatives of the log-likelihood function with
respect to the parameters. However, these derivatives are sometimes difficult
to compute with respect to the distributional parameters themselves, and are
often easier to compute with respect to an alternative (so-called 'internal')
parameterisation.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Type, Callable
from numpy import ndarray

import numpy as np
from numpy.linalg import solve


# A Value represents the 'value' of a single parameter or variate, which may in
# fact be either single-valued (i.e. scalar) or multi-valued (i.e. an array of
# different values). Here 'float' means 'float-like'.
Value = Union[float, ndarray]

# A Values instance represents the 'value(s)' of one or more parameters or variates
# (each being either single-valued or multi-valued) in some fixed order.
Values = Tuple[Value]

# A Values2D instance represents the 'value(s)' of one or more parameters or variates
# (each being either single-valued or multi-valued) in some fixed, two-dimensional order.
Values2D = Tuple[Values]


###############################################################################
# Useful functions:


def to_value(value: object) -> Value:
    """
    Converts a scalar value or array-like of values to
    the Value type.
    """
    if not isinstance(value, ndarray):
        if hasattr(value, "__len__"):
            return np.asarray(value)
        if hasattr(value, "__iter__"):
            return np.fromiter(value, float)
    return value


def is_scalar(value: Value) -> bool:
    """
    Determines whether the argument is scalar-valued or multi-valued.

    Input:
        - value (float or array-like): The input.
    Returns:
        - flag (bool): A value of True if the input is scalar, otherwise
            a value of False if the input is multi-valued.
    """
    return not isinstance(value, ndarray) or len(value.shape) == 0


def is_divergent(value: Value) -> bool:
    """
    Determines whether the value or values indicate divergence.

    Input:
        - value (float or array-like): The scalar value or
            array of values.
    Returns:
        - flag (bool): A value of True if there is any divergence,
            else False.
    """
    if is_scalar(value):
        return np.isnan(value) or np.isinf(value)
    for v in value:
        if np.isnan(v) or np.isinf(v):
            return True
    return False


def is_nondivergent_scalars(*values: Values) -> bool:
    """
    Determines whether or not the given values are all valid and scalar.

    Input:
        - values (tuple of float or ndarray): The values.
    Returns:
        - flag (bool): A value of True if all values pass, otherwise False.
    """
    for value in values:
        if not is_scalar(value):
            return False  # Multi-valued
        if np.isnan(value) or np.isinf(value):
            return False  # Divergent value
    return True


def _redimension(values: Values, n_dim: int) -> ndarray:
    """
    Redimensions the input tuple into a matrix.

    Input:
        - values (tuple of float or ndarray): The input values.
        - n_dim (int): The required dimension.

    Returns:
        - mat (ndarray): The output matrix.
    """

    def convert(value: Value) -> ndarray:
        if is_scalar(value):
            return np.array([value] * n_dim)
        if value.shape[0] != n_dim:
            raise ValueError("Incompatible dimensions!")
        return value

    return np.column_stack([convert(v) for v in values])


###############################################################################
# Base distribution class:


class ScalarPDF(ABC):
    """
    A probability distribution of a scalar variate, X.
    """

    # ----------------------------------------------------
    # Methods for specifying the distributional parameters:

    @staticmethod
    @abstractmethod
    def default_parameters() -> Values:
        """
        Provides default (scalar) values of the distributional parameters.

        Returns:
            - params (tuple of float): The default parameter values.
        """
        raise NotImplementedError

    @staticmethod
    def is_valid_parameters(*params: Values) -> bool:
        """
        Determines whether or not the given values are suitable
        distributional parameters. Specifically, there should be
        the correct number of parameters, and each parameter should
        have finite value(s) within appropriate bounds.

        Input:
            - params (tuple of float): The proposed parameter values.

        Returns:
            - flag (bool): A  value of True if the values are valid,
                else False.
        """
        # By default, just check for  divergence
        for value in params:
            if is_divergent(value):
                return False
        return True

    def __init__(self, *params: Values):
        """
        Initialises the distribution with the given parameter value(s).

        Each parameter may have either a single value or multiple values.
        If all parameters are single-valued, then only a single distribution
        is specified, and all computations, e.g. the distributional mean or
        variance, etc., will be single-valued.

        However, the use of one or more parameters with multiple values
        indicates a collection of distributions, rather than a single
        distribution. As such, all computations will be multi-valued
        rather than single-valued.

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        if len(params) == 0:
            self.set_parameters(*self.default_parameters())
        else:
            self.set_parameters(*params)

    def parameters(self) -> Values:
        """
        Provides the values of the distributional parameters.

        Returns:
            - params (tuple of float or ndarray): The parameter values.
        """
        return self._params

    def set_parameters(self, *params: Values):
        """
        Overrides the distributional parameter value(s).

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")

        _params = []
        size = 1
        for i, param in enumerate(params):
            if not is_scalar(param):
                if len(param.shape) != 1:
                    raise ValueError(f"Expected parameter {i} to be uni-dimensional")
                _len = len(param)
                if _len <= 0:
                    raise ValueError(f"Expected parameter {i} to be non-empty")
                if _len == 1:
                    # Take scalar value
                    param = param[0]
                elif size == 1:
                    size = _len
                elif size != _len:
                    raise ValueError(f"Expected parameter {i} to have length {size}")
            _params.append(param)
        self._params = tuple(_params)
        self._size = size

    # ---------------------------------------------------
    # Methods for obtaining the distributional properties:

    @abstractmethod
    def mean(self) -> Value:
        """
        Computes the mean(s) of the distribution(s).

        Returns:
            - mu (float or ndarray): The mean value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Value:
        """
        Computes the variance(s) of the distribution(s).

        Returns:
            - sigma_sq (float or ndarray): The variance(s).
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, data: Value) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - data (float or ndarray): The value(s) of the response variate.

        Returns:
            - log_prob (float or ndarray): The log-likelihood(s).
        """
        raise NotImplementedError

    # ----------------------------------------------------
    # Methods for estimating the distributional parameters:

    # Default parameters for controlling convergence of the iterative
    #   parameter estimation agorithm.
    #
    # Parameters:
    #   - max_iters (int): The maximum number of iterations allowed.
    #   - score_tol (float): The minimum difference in scores to
    #       signal convergence.
    #   - grad_tol (float): The minimum gradient size to signal
    #       convergence.
    #   - step_size (float): The parameter update scaling factor
    #       (or learning rate).
    FITTING_DEFAULTS = dict(max_iters=100, score_tol=1e-6, grad_tol=1e-6, step_size=1.0)

    def fit(
        self,
        data: Value,
        weights: Optional[Value] = None,
        init: bool = True,
        **controls: dict,
    ) -> Tuple[float, int, float]:
        """
        Estimates the distributional parameters from the given observation(s),
        by maximising the log-likelihood score.

        If the PDF does not represent a single distribution, then the paramters
        will first be reset.

        Additionally, if specified, initial parameter values will be estimated.

        Inputs:
            - data (float-like or array-like): The value(s) of the response variate.
            - weights (float-like or array-like, optional): The weight(s) of the
                observation(s).
            - init (bool, optional): Indicates whether or not to (re)initialise the
                parameter estimates; defaults to True. Set to False if the
                previous fit did not achieve convergence.
            - controls (dict): The user-specified controls. See FITTING_DEFAULTS.

        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - tol (float): The final score tolerance.
        """
        # Allow for single or multiple observations, with or without weights
        data = to_value(data)
        if weights is not None:
            weights = to_value(weights)
        if is_scalar(data):
            if weights is None:
                weights = 1.0
            elif isinstance(weights, ndarray):
                raise ValueError("Incompatible weights!")
        else:
            if weights is None:
                weights = np.ones(len(data))
            elif not isinstance(weights, ndarray) or len(weights) != len(data):
                raise ValueError("Incompatible weights!")

        # Enforce a single distribution, i.e. scalar parameter values.
        if not is_nondivergent_scalars(*self.parameters()):
            self.set_parameters(*self.default_parameters())
        if init:
            self.set_parameters(*self._estimate_parameters(data, weights, **controls))
        return self._optimise_parameters(data, weights, **controls)

    @abstractmethod
    def _estimate_parameters(
        self, data: Value, weights: Value, **kwargs: dict
    ) -> Values:
        """
        Estimates the values of the distributional parameters from
        observed data, prior to maximum likelihood estimation.

        The initial estimates will typically be approximate values,
        usually computed via the method of moments.

        Inputs:
            - data (float or ndarray): The value(s) of the response variate.
            - weights (float or ndarray): The weight(s) of the observation(s).
            - kwargs (dict, optional): Additional information, e.g. prior values.

        Returns:
            - params (tuple of float): The estimated parameter values.
        """
        raise NotImplementedError

    def _optimise_parameters(
        self, data: Value, weights: Value, **controls: dict
    ) -> Tuple[float, int, float]:
        """
        Updates the distributional parameters by maximising the
        log-likelihood score of the given observation(s).

        It is assumed that suitable initial parameter values have already been set.

        Inputs:
            - data (float or ndarray): The value(s) of the response variate.
            - weights (float or ndarray): The weight(s) of the observation(s).
            - controls (dict): The user-specified controls. See FITTING_DEFAULTS.

        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - score_tol (float): The final score tolerance.
        """
        # Create data averaging and scoring functions
        if is_scalar(data):
            # No weights or averaging required
            mean_fn = np.array
            score_fn = self.log_prob
        else:
            n_dim = data.shape[0]
            tot_weight = np.sum(weights)
            # Specify weighted mean function
            mean_fn = lambda t: (weights @ _redimension(t, n_dim)) / tot_weight
            # Specify score function
            score_fn = lambda x: (weights @ self.log_prob(x)) / tot_weight

        # Obtain the convergence controls.
        _controls = self.FITTING_DEFAULTS.copy()
        _controls.update(controls)
        max_iters = _controls["max_iters"]
        num_iters = 0
        min_grad_tol = _controls["grad_tol"]
        min_score_tol = _controls["score_tol"]
        score_tol = 0.0
        step_size = _controls["step_size"]

        # Estimate the optimal parameter values.
        score = score_fn(data)
        alt_params = np.array(
            self._internal_parameters(*self.parameters()), dtype=float
        )

        while num_iters < max_iters:
            # Check if gradient is close to zero
            g = mean_fn(self._internal_gradient(data))
            if np.max(np.abs(g)) < min_grad_tol:
                break

            # Find Newton-Raphson update
            num_iters += 1
            n_hess = self._internal_neg_hessian(data)
            n_hess = np.array([mean_fn(r) for r in self._internal_neg_hessian(data)])
            d_alt_params = solve(n_hess, g)

            # Apply line search
            _step_size = step_size
            while True:
                try:
                    alt_params += _step_size * d_alt_params
                    self.set_parameters(*self._distributional_parameters(*alt_params))
                    break
                except:
                    alt_params -= _step_size * d_alt_params
                    _step_size *= 0.5

            # Obtain new score
            new_score = score_fn(data)
            score_tol = new_score - score
            score = new_score
            if np.abs(score_tol) < min_score_tol:
                break

        return score, num_iters, score_tol

    def _internal_parameters(self, *params: Values) -> Values:
        """
        Converts distributional parameter values into an internal
        representation suitable for maximum likelihood estimation.

        Input:
            - params (tuple of float or ndarray): The distributional parameter values.

        Returns:
            - alt_params (tuple of float or ndarray): The internal parameter values.
        """
        # By default, assume an identity transformation
        return params

    def _distributional_parameters(self, *alt_params: Values) -> Values:
        """
        Converts internal parameter values into their standard
        distributional representation.

        Input:
            - alt_params (tuple of float or ndarray): The internal parameter values.

        Returns:
            - params (tuple of float or ndarray): The distributional parameter values
        """
        # By default, assume an identity transformation
        return alt_params

    @abstractmethod
    def _internal_gradient(self, data: Value) -> Values:
        """
        Computes the gradient of the log-likelihood function
        with respect to the internal parameterisation, evaluated
        at the observed value(s).

        Input:
            - data (float or ndarray): The value(s) of the response variate.

        Returns:
            - grad (tuple of float or ndarray): The parameter gradients.
        """
        raise NotImplementedError

    @abstractmethod
    def _internal_neg_hessian(self, data: Value) -> Values2D:
        """
        Computes the negative of the Hessian matrix of second derivatives
        of the log-likelihood function with respect to the internal
        parameterisation, evaluated at the observed value(s).

        Note that the expected value of the negative Hessian matrix is the
        variance matrix of the internal variates. Consequently, the returned
        matrix is allowed to be an approximation, provided that the iterative
        updates still converge.

        Input:
            - data (float or ndarray): The value(s) of the response variate.

        Returns:
            - Sigma (matrix of float or ndarray): The second derivatives.
        """
        raise NotImplementedError


###############################################################################
# More helper methods:


def fit_defaults(**controls: dict) -> Callable[[Type[ScalarPDF]], Type[ScalarPDF]]:
    """
    Modifies the default values of the convergence controls for
    the fit() parameter estimation algorithm.

    Input:
        - controls (dict): The overriden controls and their default values.
            See FITTING_DEFAULTS.
    """

    def decorator(klass: Type[ScalarPDF]) -> Type[ScalarPDF]:
        _controls = klass.FITTING_DEFAULTS.copy()
        _controls.update(controls)
        klass.FITTING_DEFAULTS = _controls
        return klass

    return decorator


def check_transformations(pdf: ScalarPDF) -> bool:
    """
    Checks whether or not the dual transformations between
    distributional parameters and internal parameters are
    consistent.

    Input:
        - pdf (ScalarPDF): A PDF instance.

    Returns:
        - flag (bool): A value of True if the transformations
            are consistent, else False.
    """
    params = pdf.parameters()
    alt_params = pdf._internal_parameters(*params)
    params2 = pdf._distributional_parameters(*alt_params)
    alt_params2 = pdf._internal_parameters(*params2)
    for v1, v2 in zip(params, params2):
        if np.abs(v1 - v2) >= 1e-6:
            return False
    for v1, v2 in zip(alt_params, alt_params2):
        if np.abs(v1 - v2) >= 1e-6:
            return False
    return True
