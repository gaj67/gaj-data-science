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

import numpy as np
from numpy.linalg import solve

from .value_types import (
    Value,
    ValueLike,
    Values,
    Values2d,
    Vector,
    is_scalars,
    to_vector,
    to_matrix,
)

from .distribution import Parameterised


###############################################################################
# Useful types:


# Encapsulates any and all settings necessary to control the
# parameter estimation algorithm. See Fitter.default_controls().
Controls: TypeAlias = Dict[str, Any]

# Encapsulates the performance summary of the parameter estimation algorithm.
Results: TypeAlias = Dict[str, Any]


###############################################################################
# Useful functions:


def mean_value(weights: Vector, value: Value) -> float:
    """
    Computes the weighted mean of the scalar or vector value.

    Input:
        - weights (ndarray): The vector of weights.
        - values (float-like or vector): The scalar or vector value.

    Returns:
        - mean (float): The value mean.
    """
    return (weights @ to_vector(value)) / np.sum(weights)


def mean_values(weights: Vector, values: Values) -> Vector:
    """
    Computes the weighted mean of the scalar or vector values.

    Input:
        - weights (ndarray): The vector of weights.
        - values (tuple of float-like or vector): The scalar or vector values.

    Returns:
        - means (ndarray): The vector of value means.
    """
    return (weights @ to_matrix(values, len(weights))) / np.sum(weights)


###############################################################################
# Abstract data fitting class:


class Transformable:
    """
    Provides an invertable transformation between the default parameterisation
    and an alternative parameterisation.
    """

    def transform(self, *params: Values) -> Values:
        """
        Transforms the parameter values into an alternative representation.

        Input:
            - params (tuple of float-like or vector): The parameter values.

        Returns:
            - alt_params (tuple of float-like or vector): The alternative parameter values.
        """
        # By default, assume an identity transformation
        return params

    def inverse_transform(self, *alt_params: Values) -> Values:
        """
        Transforms the alternative parameter values into the usual representation.

        Input:
            - alt_params (tuple of float-like or vector): The alternative parameter values.

        Returns:
            - params (tuple of float-like or vector): The parameter values
        """
        # By default, assume an identity transformation
        return alt_params


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

    def underlying(self) -> Parameterised:
        """
        Obtains the underlying parameterisation.

        Returns:
            - params (Parameterised): The parameterisation.
        """
        return self._params

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
        data: ValueLike,
        weights: Optional[ValueLike] = None,
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
            self.underlying().set_parameters(
                *self.estimate_parameters(v_data, v_weights, all_controls)
            )
        elif not is_scalars(*self.underlying().parameters()):
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
            - data (float-like or vector): The value(s) of the observation(s).
            - weights (float-like or vector): The weight(s) of the observation(s).
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - score_tol (float): The final score tolerance.
        """
        # Get current parameter estimates
        params = self.underlying().parameters()
        print("DEBUG[0]: Initial parameters =", params)
        score = mean_value(weights, self.compute_score(params, data, controls))
        score_tol = 0.0

        tf = self if isinstance(self, Transformable) else Transformable()

        num_iters = 0
        while num_iters < controls["max_iters"]:
            # Obtain update and check if small
            delta_params = self.compute_update(params, data, weights, controls)
            print("DEBUG: update =", delta_params)
            if np.max(np.abs(delta_params)) < controls["grad_tol"]:
                break

            # Apply line search
            num_iters += 1
            vec_params = np.array(tf.transform(*params), dtype=float)
            step_size = controls["step_size"]
            while True:
                # Apply update
                vec_params += step_size * delta_params
                params = tf.inverse_transform(*vec_params)
                if self.underlying().is_valid_parameters(*params):
                    break
                # Retract update
                vec_params -= step_size * delta_params
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
            self.underlying().set_parameters(*params)

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

        NOTE: If the fitter is Transformable then the update is with respect to
        the transformed parameterisation.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - weights (ndarray): The weight(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - delta (ndarray): The parameter update vector.
        """
        raise NotImplementedError


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
        data: ValueLike,
        weights: Optional[ValueLike] = None,
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
            data: ValueLike,
            weights: Optional[ValueLike] = None,
            **controls: Controls,
        ) -> Results:
            return fitter_class(self).fit(data, weights, **controls)

        klass.fit = fit
        klass.fit.__doc__ = Fittable.fit.__doc__
        return klass

    return decorator


###############################################################################
# Specialised data fitting classes:


@fitting_controls(step_size=0.1)
class GradientFitter(Fitter):
    """
    Estimates the parameters of the implemented objective function
    via iterative gradient optimisation.

    The parameter values are stored in a Parameterised instance.
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

        NOTE: If the fitter is Transformable then derivatives are with respect to
        the transformed parameterisation.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - grad (tuple of float-like or vector): The first derivatives.
        """
        raise NotImplementedError


class NewtonRaphsonFitter(Fitter):
    """
    Estimates the parameters of the implemented objective function
    via iterative Newton-Raphson optimisation.

    The parameter values are stored in a Parameterised instance.
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

        NOTE: If the fitter is Transformable then derivatives are with respect to
        the transformed parameterisation.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - grad (tuple of float-like or vector): The first derivatives.
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

        NOTE: If the fitter is Transformable then derivatives are with respect to
        the transformed parameterisation.

        Note that the expected value of the negative Hessian matrix is the
        variance matrix of the alternative variates. Consequently, the returned
        matrix is allowed to be an approximation, provided that the iterative
        updates still converge.

        Input:
            - params (tuple of float): The current parameter estimatess.
            - data (ndarray): The value(s) of the observation(s).
            - controls (dict): Additional user-supplied information.

        Returns:
            - Sigma (matrix of float-like or vector): The negative second derivatives.
        """
        raise NotImplementedError
