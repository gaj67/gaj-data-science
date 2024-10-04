"""
This module defines the base class for a probability distribution function (PDF)
of a (discrete or continuous) scalar variate, in terms of one or more
distributional parameters.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Type, Callable
from numpy import ndarray

import numpy as np
from numpy.linalg import solve


"""
A Value represents the 'value' of a single parameter or variate, which may in
fact be either single-valued (i.e. scalar) or multi-valued (i.e. an array of
different values).
"""
Value = Union[float, ndarray]

"""
A Values instance represents the 'value(s)' of one or more parameters or variates
(each being either single-valued or multi-valued) in some fixed order.
"""
Values = Tuple[Value]

"""
A Values2D instance represents the 'value(s)' of one or more parameters or variates
(each being either single-valued or multi-valued) in some fixed, two-dimensional order.
"""
Values2D = Tuple[Values]


###############################################################################
# Useful functions:


def is_multi(X: Value) -> bool:
    """
    Determines whether the input is multi-valued or single-valued.

    Input:
        - X (float-like or array-like): The input.
    Returns:
        - flag (bool): A value of True if the input is multi-valued, otherwise
            a value of False.
    """
    return hasattr(X, "__len__") or hasattr(X, "__iter__")


def is_scalar(*params: Values) -> bool:
    """
    Determines whether or not the given parameters all have valid, scalar
    values.

    Input:
        - params (tuple of float or ndarray): The parameter values.
    Returns:
        - flag (bool): A value of True if scalar-valued, otherwise False.
    """
    for p in params:
        if is_multi(p):
            return False  # Multi-valued
        if np.isnan(p):
            return False  # Divergent parameters
    return True


###############################################################################
# Base distribution class:


class ScalarPDF(ABC):
    """
    A probability distribution of a scalar variate, X.
    """

    # ----------------------------------------------------
    # Methods for specifying the distributional parameters:

    def __init__(self, *theta: Values):
        """
        Initialises the distribution with the given parameter values.

        Each parameter may have either a single value or multiple values.
        If all parameters are single-valued, then only a single distribution
        is specified, and all computations, e.g. the distributional mean or
        variance, etc., will be single-valued.

        However, the use of one or more parameters with multiple values
        indicates a collection of distributions, rather than a single
        distribution. As such, all computations will be multi-valued
        rather than single-valued.

        Input:
            - theta (tuple of float or ndarray): The parameter value(s).
        """
        if len(theta) > 0:
            self.set_parameters(*theta)
        else:
            self.reset_parameters()

    def parameters(self) -> Values:
        """
        Provides the values of the distributional parameters.

        Returns:
            - theta (tuple of float or ndarray): The parameter values.
        """
        return self._params

    def set_parameters(self, *theta: Values):
        """
        Initialises the distributional parameter value(s).

        Input:
            - theta (tuple of float or ndarray): The parameter value(s).
        """
        params = []
        size = 1
        for i, param in enumerate(theta):
            if isinstance(param, ndarray):
                if len(param.shape) != 1:
                    raise ValueError(f"Expected parameter {i} to be uni-dimensional")
                _len = len(param)
                if _len <= 0:
                    raise ValueError(f"Expected parameter {i} to be non-empty")
                elif _len == 1:
                    # Take scalar value
                    param = param[0]
                elif size == 1:
                    size = _len
                elif size != _len:
                    raise ValueError(f"Expected parameter {i} to have length {size}")
            params.append(param)
        self._params = tuple(params)
        self._size = size

    @abstractmethod
    def reset_parameters(self):
        """
        Resets the distributional parameters to their default (scalar) values.
        """
        raise NotImplementedError

    # ---------------------------------------------------
    # Methods for obtaining the distributional properties:

    def __len__(self):
        """
        Determines the number of distributions represented by this instance.

        Returns:
            - length (int): The number of distributions.
        """
        return self._size

    def is_scalar(self) -> bool:
        """
        Determines whether or not the distribution has valid, scalar-valued
        parameters.

        Returns:
            - flag (bool): A value of True if scalar-valued, otherwise False.
        """
        if len(self) > 1:
            return False  # Multi-valued
        return is_scalar(*self.parameters())

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
    def log_prob(self, X: Value) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - X (float or ndarray): The value(s) of the response variate.

        Returns:
            - log_prob (float or ndarray): The log-likelihood(s).
        """
        raise NotImplementedError

    # ----------------------------------------------------
    # Methods for estimating the distributional parameters:

    """
    Default parameters for controlling convergence of the iterative
    parameter estimation agorithm.
    
    Parameters:
        - max_iters (int): The maximum number of iterations allowed.
        - score_tol (float): The minimum difference in scores to 
            signal convergence.
        - grad_tol (float): The minimum gradient size to signal
            convergence.
        - step_size (float): The parameter update scaling factor
            (or learning rate).
    """
    FITTING_DEFAULTS = dict(
        max_iters=100, score_tol=1e-6, grad_tol=1e-6, step_size=1.0
    )


    @abstractmethod
    def initialise_parameters(self, X: Value, W: Optional[Value] = None, **kwargs: dict):
        """
        Initialises the distributional parameter values prior to
        maximum likelihood estimation.

        The initial estimates will typically be approximate values,
        usually computed via the method of moments.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - W (float or ndarray, optional): The weight(s) of the observation(s).
            - kwargs (dict, optional): Additional information, e.g. prior values.
        """
        raise NotImplementedError

    def estimate_parameters(
        self, X: Value, W: Optional[Value] = None, **controls: dict
    ) -> Tuple[float, int, float]:
        """
        Estimates the distributional parameters from the given observation(s),
        by maximising the log-likelihood score.

        It is assumed that suitable initial parameter values have already been set.
        
        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - W (float or ndarray, optional): The weight(s) of the observation(s).
            - controls (dict): The user-specified controls. See FITTING_DEFAULTS.

        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - score_tol (float): The final score tolerance.
        """
        # Allow for single or multiple observations
        if is_multi(X) and not isinstance(X, ndarray):
            X = np.fromiter(X, float)

        # Create data averaging and scoring functions
        if isinstance(X, ndarray):
            # Obtain a weight for each observation
            if W is None:
                W = np.ones(len(X))
            elif not isinstance(W, ndarray) or len(W) != len(X):
                raise ValueError("Incompatible weights!")
            tot_W = np.sum(W)
            # Specify weighted mean function
            mean_fn = lambda t: (W @ np.column_stack(t)) / tot_W
            # Specify score function
            score_fn = lambda x: (W @ self.log_prob(x)) / tot_W
        else:
            # No weights or averaging required
            mean_fn = np.array
            score_fn = self.log_prob

        # Obtain the convergence controls.
        _controls = self.FITTING_DEFAULTS.copy()
        _controls.update(controls)
        print("DEBUG: controls =", _controls)
        max_iters = _controls["max_iters"]
        num_iters = 0
        min_grad_tol = _controls["grad_tol"]
        min_score_tol = _controls["score_tol"]
        score_tol = 0.0
        step_size = _controls["step_size"]

        # Estimate the optimal parameter values.
        score = score_fn(X)
        theta = np.array(self.parameters(), dtype=float)

        while num_iters < max_iters:
            # Check if gradient is close to zero
            g = mean_fn(self.gradient(X))
            if np.min(np.abs(g)) < min_grad_tol:
                break

            # Apply Newton-Raphson update
            num_iters += 1
            nH = np.array([mean_fn(r) for r in self.negative_Hessian(X)])
            d_theta = solve(nH, g)
            theta += step_size * d_thets

            # Obtain new score
            new_score = score_fn(X)
            score_tol = new_score - score
            score = new_score
            print(
                "DEBUG: num_iters =", num_iters, "tol =", tol, "\n", self.parameters()
            )
            if np.abs(score_tol) < min_score_tol:
                break

        return score, num_iters, score_tol

    def fit(
        self, X: Value, W: Optional[Value] = None, init: bool = True, **controls: dict
    ) -> Tuple[float, int, float]:
        """
        Estimates the distributional parameters from the given observation(s),
        by maximising the log-likelihood score.

        If the PDF does not represent a single distribution, then the paramters
        will first be reset.

        Additionally, if specified, initial parameter values will be estimated.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - W (float or ndarray, optional): The weight(s) of the observation(s).
            - init (bool, optional): Indicates whether or not to (re)initialise the
                parameter estimates; defaults to True. Set to False if the
                previous fit did not achieve convergence.
            - controls (dict): The user-specified controls. See FITTING_DEFAULTS.

        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - tol (float): The final score tolerance.
        """
        # Enforce a single distribution, i.e. scalar parameter values.
        if not self.is_scalar():
            self.reset_parameters()
        if init:
            self.initialise_parameters(X, W, **controls)
        return self.estimate_parameters(X, W, **controls)

    @abstractmethod
    def gradient(self, X: Value) -> Values:
        """
        Computes the gradient of the log-likelihood function
        with respect to the distributional parameters, evaluated
        at the observed value(s).
        
        Input:
            - X (float or ndarray): The value(s) of the response variate.

        Returns:
            - grad (tuple of float or ndarray): The parameter gradients.
        """
        raise NotImplementedError

    @abstractmethod
    def negative_Hessian(self, X: Value) -> Values2D:
        """
        Computes the negative of the Hessian matrix of second derivatives
        of the log-likelihood function with respect to the distributional
        parameters, evaluated at the observed value(s).

        Note that the expected value of the negative Hessian matrix is the
        variance matrix. Consequently, the returned matrix is allowed to be
        an approximation, provided that the iterative updates still converge.
        
        Input:
            - X (float or ndarray): The value(s) of the response variate.

        Returns:
            - Sigma (matrix of float or ndarray): The parameter variations.
        """
        raise NotImplementedError


###############################################################################
# Class decorator:


def FittingControls(**controls: dict) -> Callable[[Type[ScalarPDF]], Type[ScalarPDF]]:
    """
    Modifies the default values of the convergence controls for
    the parameter estimation algorithm.

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
