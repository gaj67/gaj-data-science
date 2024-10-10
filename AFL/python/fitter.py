"""
This module defines the base classes for algorithms to fit observed data,
i.e. estimate parameters from data.

It is assumed that some parameterised objective function will be optimised. 

However, for convenience, any computed  derivatives of the objective function
may be with respect to an alternative parameterisation rather than the
underlying, direct parameterisation.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, Optional, Type, Callable, Dict, Any
from numpy import ndarray

import numpy as np
from numpy.linalg import solve

from distribution import (
    Parameterised,
    Value,
    Values,
    Values2d,
    Vector,
    is_scalar,
)


###############################################################################
# Useful types:


# Encapsulates any and all settings necessary to control the
# parameter estimation algorithm. See Fitter.default_controls().
Controls: TypeAlias = Dict[str, Any]

# Encapsulates the performance summary of the parameter estimation algorithm.
Results: TypeAlias = Dict[str, Any]

# A Matrix is a 2D array
Matrix: TypeAlias = ndarray


###############################################################################
# Useful functions:


def is_scalars(*values: Values) -> bool:
    """
    Determines whether or not the given values are all scalar.

    Input:
        - values (tuple of float or ndarray): The values.
    Returns:
        - flag (bool): A value of True if all values are scalar, otherwise False.
    """
    return np.all(list(map(is_scalar, values)))


def to_vector(value: object) -> Vector:
    """
    Converts the value(s) to an ndarray vector.

    Input:
        - value (float-like or array-like): The input value(s).

    Returns:
        - vec (ndarray): The output vector.
    """
    if isinstance(value, ndarray):
        if len(value.shape) == 0:
            vec = np.array([value])
        else:
            vec = value
    elif hasattr(value, "__len__"):
        vec = np.asarray(value)
    elif hasattr(value, "__iter__"):
        vec = np.fromiter(value, float)
    else:
        vec = np.array([value])
    if len(vec.shape) != 1:
        raise ValueError("Value must be scalar or vector-like")
    return vec


def to_matrix(values: Values, n_dim: int) -> Matrix:
    """
    Redimensions the input tuple into a matrix.

    Input:
        - values (tuple of float or ndarray): The input values.
        - n_dim (int): The required row dimension.

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


def mean_value(weights: Vector, value: Value) -> float:
    """
    Computes the weighted mean of the scalar or vector value.

    Input:
        - weights (ndarray): The vector of weights.
        - values (float or ndarray): The scalar or vector value.

    Returns:
        - mean (float): The value mean.
    """
    return (weights @ to_vector(value)) / np.sum(weights)


def mean_values(weights: Vector, values: Values) -> Vector:
    """
    Computes the weighted mean of the scalar or vector values.

    Input:
        - weights (ndarray): The vector of weights.
        - values (tuple of float or ndarray): The scalar or vector values.

    Returns:
        - means (ndarray): The vector of value means.
    """
    return (weights @ to_matrix(values, len(weights))) / np.sum(weights)


###############################################################################
# Abstract data fitting class:


class Fitter(ABC):
    """
    Estimates the parameters of a supplied Parameterised instance
    via iterative optimisation of an objective function.

    For convenience, the value (or score) of the objective function
    (and the values of any derivatives) will be computed with respect
    to an alternative, invertable parameterisation.
    """

    # ----------------------------------------------------------------
    # Methods for the Parameterised instance

    def __init__(self, params: Parameterised):
        """
        Wraps the parameters to be optimised with the data fitter.

        Input:
            - params (Parameterised): The parameters holder.
        """
        self._params = params

    def get_parameters(self) -> Values:
        """
        Obtains the current values of the parameters.

        Returns:
            - params (Values): The parameter values.
        """
        return self._params.parameters()

    def is_valid_parameters(self, *params: Values) -> bool:
        """
        Checks whether the current estimates of the underlying
        parameters are valid.

        Input:
            - params (Values): The parameter values.

        Returns:
            - flag (bool): A  value of True if the values are valid,
                else False.
        """
        return self._params.is_valid_parameters(*params)

    def set_parameters(self, *params: Values):
        """
        Overrides the values of the parameters.

        Input:
            - params (Values): The parameter values.
        """
        return self._params.set_parameters(*params)

    # ----------------------------------------------------------------
    # Methods for the estimation algorithm

    @staticmethod
    def default_controls() -> Controls:
        """
        Provides default settings for controlling convergence of the fit() algorithm.

        Parameters:
            - init (bool): Indicates whether or not to (re)initialise the
                parameter estimates; defaults to True. Set to False if the
                previous fit did not achieve convergence.
            - max_iters (int): The maximum number of iterations allowed.
            - score_tol (float): The minimum difference in scores to signal convergence.
            - grad_tol (float): The minimum gradient size to signal convergence.
            - step_size (float): The parameter update scaling factor (or learning rate).
        """
        return {
            "init": True,
            "max_iters": 100,
            "score_tol": 1e-6,
            "grad_tol": 1e-6,
            "step_size": 1.0,
        }

    def fit(
        self,
        data: Value,
        weights: Optional[Value] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - data (float-like or array-like): The value(s) of the observation(s).
            - weights (float-like or array-like, optional): The weight(s) of the
                observation(s).
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output. See optimise_parameters().
        """
        # Allow for single or multiple observations, with or without weights
        v_data = to_vector(data)
        if weights is None:
            v_weights = np.ones(len(v_data))
        else:
            v_weights = to_vector(weights)
            if len(v_weights) != len(v_data):
                raise ValueError("Incompatible weights!")

        # Obtains controls
        all_controls = self.default_controls()
        all_controls.update(controls)
        print("DEBUG: controls =", all_controls)

        # Enforce a single distribution, i.e. scalar parameter values.
        if all_controls["init"]:
            self.set_parameters(
                *self.estimate_parameters(v_data, v_weights, all_controls)
            )
        elif not is_scalars(*self.get_parameters()):
            raise ValueError("Parameters are multi-valued!")

        # Iteratively optimise the parameters
        return self.optimise_parameters(v_data, v_weights, all_controls)

    @abstractmethod
    def estimate_parameters(
        self, data: Vector, weights: Vector, controls: Controls
    ) -> Values:
        """
        Estimates the values of the parameters from observed
        data, prior to iterative re-estimation.

        The initial estimates will typically be approximate values,
        usually computed via the method of moments.

        Inputs:
            - data (ndarray): The value(s) of the observation(s).
            - weights (ndarray): The weight(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - params (tuple of float): The estimated parameter values.
        """
        raise NotImplementedError

    def optimise_parameters(
        self, data: Vector, weights: Vector, controls: Controls
    ) -> Results:
        """
        Updates the parameters by optimising the mean score of the
        objective function for the given observation(s).

        It is assumed that suitable initial parameter values have already been set.

        Inputs:
            - data (float or ndarray): The value(s) of the observation(s).
            - weights (float or ndarray): The weight(s) of the observation(s).
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - score_tol (float): The final score tolerance.
        """
        # Get current parameter estimates
        params = self.get_parameters()
        print("DEBUG[0]: Initial parameters =", params)
        score = mean_value(weights, self.compute_score(params, data, controls))
        score_tol = 0.0

        num_iters = 0
        while num_iters < controls["max_iters"]:
            # Obtain update and check if small
            d_alt_params = self.compute_update(params, data, weights, controls)
            print("DEBUG: update =", d_alt_params)
            if np.max(np.abs(d_alt_params)) < controls["grad_tol"]:
                break

            # Apply line search
            num_iters += 1
            alt_params = np.array(self.transform_parameters(*params), dtype=float)
            step_size = controls["step_size"]
            while True:
                # Apply update
                alt_params += step_size * d_alt_params
                params = self.invert_parameters(*alt_params)
                if self.is_valid_parameters(*params):
                    break
                # Retract update
                alt_params -= step_size * d_alt_params
                # Reduce step-size
                step_size *= 0.5

            # Obtain new score and check convergence
            new_score = mean_value(weights, self.compute_score(params, data, controls))
            score_tol = new_score - score
            score = new_score
            print(f"DEBUG[{num_iters}]: score={score}, new parameters =", params)
            if np.abs(score_tol) < controls["score_tol"]:
                break

        if num_iters > 0:
            print("DEBUG: Final parameters =", params)
            self.set_parameters(*params)

        return {
            "score": score,
            "score_tol": score_tol,
            "num_iters": num_iters,
        }

    @abstractmethod
    def compute_score(self, params: Values, data: Vector, controls: Controls) -> Value:
        """
        Computes the score(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - score (ndarray): The objective function score(s).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_update(
        self, params: Values, data: Vector, weights: Vector, controls: Controls
    ) -> Vector:
        """
        Computes a parameter update vector in the direction of improving
        the objective function, evaluated at the current parameter estimates
        and averaged over the observed value(s).

        NOTE: Derivatives are with respect to the alternative parameterisation.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - weights (ndarray): The weight(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - delta (ndarray): The alternative parameter update vector.
        """
        raise NotImplementedError

    def transform_parameters(self, *params: Values) -> Values:
        """
        Transforms the parameter values into an internal representation
        more suitable for optimisation.

        Input:
            - params (tuple of float or ndarray): The parameter values.

        Returns:
            - alt_params (tuple of float or ndarray): The internal parameter values.
        """
        # By default, assume an identity transformation
        return params

    def invert_parameters(self, *alt_params: Values) -> Values:
        """
        Transforms the internal parameter values into the usual representation.

        Input:
            - alt_params (tuple of float or ndarray): The internal parameter values.

        Returns:
            - params (tuple of float or ndarray): The parameter values
        """
        # By default, assume an identity transformation
        return alt_params


# Decorator for easily overriding the default values of fitting controls
def fitting_controls(**controls: Controls) -> Callable[[Type[Fitter]], Type[Fitter]]:
    """
    Modifies the default values of the convergence controls for the fit()
    parameter estimation algorithm.

    Input:
        - controls (dict): The overriden controls and their new default values.
            See Fitter.default_controls().

    Returns:
        - decorator (method): A decorator of a Fitter class.
    """

    def decorator(klass: Type[Fitter]) -> Type[Fitter]:
        default_controls_fn = klass.default_controls

        @staticmethod
        def default_controls() -> Controls:
            new_controls = default_controls_fn()
            new_controls.update(controls)
            return new_controls

        klass.default_controls = default_controls
        klass.default_controls.__doc__ = default_controls_fn.__doc__
        return klass

    return decorator


###############################################################################
# Abstract data fitting interface:

# NOTE: Don't make Fittable subclass ABC otherwise overriding fitter() method
# via @add_fitter() doesn't remove the abstraction.


class Fittable:
    """
    Provides an interface for estimating parameters from data.
    """

    @abstractmethod
    def fit(
        self,
        data: Value,
        weights: Optional[Value] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - data (float-like or array-like): The value(s) of the observation(s).
            - weights (float-like or array-like, optional): The weight(s) of the
                observation(s).
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output. See optimise_parameters().
        """
        raise NotImplementedError


# Decorator for easily adding a fitter() implementation
def add_fitter(
    fitter_class: Type[Fitter],
) -> Callable[[Type[Parameterised]], Type[Fittable]]:
    """
    Implements the Fittable.fitter() method to wrap a Parameterised instance
    with an instance of the specified Fitter class.

    Input:
        - fitter_class (class): The class of a Fitter implementation.

    Returns:
        - decorator (method): A decorator of a Parameterised & Fittable class.

    """

    def decorator(klass: Type[Parameterised]) -> Type[Fittable]:

        def fit(
            self: Parameterised,
            data: Value,
            weights: Optional[Value] = None,
            **controls: Controls,
        ) -> Results:
            return fitter_class(self).fit(data, weights, **controls)

        klass.fit = fit
        klass.fit.__doc__ = Fittable.fit.__doc__
        return klass

    return decorator


###############################################################################
# Specialised data fitting classes:


class GradientFitter(Fitter):
    """
    Estimates the parameters of the implemented objective function
    via iterative gradient optimisation.

    The parameter values are stored in a Parameterised instance.

    For convenience, derivatives of the objective function may be
    computed with respect to an alternative parameterisation.
    """

    def compute_update(
        self, params: Values, data: Vector, weights: Vector, controls: Controls
    ) -> Vector:
        return mean_values(weights, self.compute_gradient(params, data, controls))

    @abstractmethod
    def compute_gradient(
        self, params: Values, data: Vector, controls: Controls
    ) -> Values:
        """
        Computes the gradient(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        NOTE: Derivatives are with respect to the alternative parameterisation.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - grad (tuple of float or ndarray): The first derivatives
                with respect to the alternative parameterisation.
        """
        raise NotImplementedError


class NewtonRaphsonFitter(Fitter):
    """
    Estimates the parameters of the implemented objective function
    via iterative Newton-Raphson optimisation.

    The parameter values are stored in a Parameterised instance.

    For convenience, derivatives of the objective function may be
    computed with respect to an alternative parameterisation.
    """

    def compute_update(
        self, params: Values, data: Vector, weights: Vector, controls: Controls
    ) -> Vector:
        grad = mean_values(weights, self.compute_gradient(params, data, controls))
        n_hess = np.array(
            [
                mean_values(weights, r)
                for r in self.compute_neg_hessian(params, data, controls)
            ]
        )
        return solve(n_hess, grad)

    @abstractmethod
    def compute_gradient(
        self, params: Values, data: Vector, controls: Controls
    ) -> Values:
        """
        Computes the gradient(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        NOTE: Derivatives are with respect to the alternative parameterisation.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - grad (tuple of float or ndarray): The first derivatives
                with respect to the alternative parameterisation.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_neg_hessian(
        self, params: Values, data: Vector, controls: Controls
    ) -> Values2d:
        """
        Computes the negative of the Hessian matrix of second derivatives
        of the objective function for the current parameter estimates,
        evaluated at the observed value(s).

        NOTE: Derivatives are with respect to the alternative parameterisation.

        Note that the expected value of the negative Hessian matrix is the
        variance matrix of the alternative variates. Consequently, the returned
        matrix is allowed to be an approximation, provided that the iterative
        updates still converge.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - Sigma (matrix of float or ndarray): The negative second derivatives
                with respect to the alternative parameterisation.
        """
        raise NotImplementedError
