"""
This module defines the base classes for algorithms to conditionlly fit
observed data, i.e. estimate parameters.

It is assumed that some parameterised objective function will be optimised. 
"""

from abc import ABC, abstractmethod
from typing import Tuple, Type, Callable, TypeAlias, Dict, Any, NamedTuple, Optional

import numpy as np
from numpy.linalg import solve

from .data_types import (
    Values,
    Values2d,
    Vector,
    Matrix,
    mean_value,
    mean_values,
)

from .distribution import Parameterised


###############################################################################
# Useful data-types and methods:

# Encapsulates any and all settings necessary to control the
# parameter estimation algorithm. See Fitter.default_controls().
Controls: TypeAlias = Dict[str, Any]

# Encapsulates the performance summary of the parameter estimation algorithm.
Results: TypeAlias = Dict[str, Any]


class Data(NamedTuple):
    """
    Provides a wrapper for observed data.
    """

    variate: Vector
    weights: Vector
    covariates: Optional[Matrix] = None


def diff_tolerance(values1: Values, values2: Values) -> float:
    """
    Computes the maximum absolute difference between two sequences
    of comparable values.

    Input:
        - values1 (tuple of floaat or vector): The initial values.
        - values2 (tuple of floaat or vector): The final values.

    Returns:
        - tol (float): The tolerance between the values.
    """
    return max(np.max(np.abs(v1 - v2)) for v1, v2 in zip(values1, values2))


###############################################################################
# Abstract data fitting classes:


class Optimiser(ABC):
    """
    Estimates the parameters of a supplied instance
    via iterative optimisation of an objective function.
    """

    # ----------------------------------------------------------------
    # Methods for the parameters

    def __init__(self, store: Parameterised):
        """
        Wraps the parameters to be optimised with a data fitter.

        Input:
            - store (parameterised): The owner of the parameters.
        """
        self._store = store

    def storage(self) -> Parameterised:
        """
        Obtains the underlying parameter storage.

        Returns:
            - store (parameterised): The owner of the parameters.
        """
        return self._store

    # ----------------------------------------------------------------
    # Defaults for controlling the estimation algorithm

    @staticmethod
    def default_controls() -> Controls:
        """
        Provides default settings for controlling convergence of the optimisation algorithm.

        Parameters:
            - init (bool): Indicates whether or not to (re)initialise the
                parameter estimates; defaults to True. Set to False if the
                previous fit did not achieve convergence.
            - max_iters (int): The maximum number of iterations allowed.
            - step_iters (int): The maximum number of step-size line searches allowed.
            - score_tol (float): The minimum change in score to signal convergence.
            - param_tol (float): The minimum change in parameter values to signal convergence.
            - step_size (float): The parameter update scaling factor (or learning rate).
        """
        return {
            "init": True,
            "max_iters": 100,
            "step_iters": 10,
            "score_tol": 1e-8,
            "param_tol": 1e-6,
            "step_size": 1.0,
        }

    # ----------------------------------------------------------------
    # Methods for the estimation algorithm

    def fit(
        self,
        data: Data,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output. See optimise_parameters().
        """
        # Obtain specified controls
        all_controls = self.default_controls()
        all_controls.update(controls)

        # Initialise parameter values.
        if all_controls["init"]:
            params = self.estimate_parameters(data, all_controls)
            self.storage().set_parameters(*params)
        else:
            params = self.storage().parameters()

        # Iteratively optimise the parameters
        params, results = self.optimise_parameters(params, data, all_controls)
        self.storage().set_parameters(*params)
        return results

    @abstractmethod
    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        """
        Estimates initial values of the parameters from observed data,
        prior to re-estimation.

        The initial estimates will typically be approximate values,
        usually computed via the method of moments.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - params (tuple of float or vector): The estimated values of the
                parameters.
        """
        raise NotImplementedError

    def optimise_parameters(
        self, params: Values, data: Data, controls: Controls
    ) -> Tuple[Values, Results]:
        """
        Iteratively estimates the values of the parameters by optimising
        the mean score of the objective function for the given observations.

        Inputs:
            - params (tuple of float or vector): The initial values of the parameters.
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - params (tuple of float or vector): The estimated values of the parameters.
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - score_tol (float): The final score-change tolerance.
                - param_tol (float): The final parameter-change tolerance.
                - converged (bool): Indicates whether or not parameter
                    or score convergence was achieved.
        """
        # Get score of current parameters
        score = self.compute_score(params, data, controls)
        print("DEBUG[optimise_parameters]: score =", score)
        score_tol = 0.0
        param_tol = 0.0
        num_iters = 0
        converged = False

        while num_iters < controls["max_iters"]:
            # Obtain update
            d_params = self.compute_update(params, data, controls)
            print("DEBUG[optimise_parameters]: d_params =", d_params)

            # Apply line search
            num_iters += 1
            sub_iters = 0
            step_size = controls["step_size"]
            while True:
                # Limit looping
                sub_iters += 1
                if sub_iters > controls["step_iters"]:
                    raise ValueError("Parameters failed to converge within bounds!")
                # Apply update
                new_params = tuple(v + step_size * d for v, d in zip(params, d_params))
                print("DEBUG[optimise_parameters]: new_params =", new_params)
                if self.storage().is_valid_parameters(*new_params):
                    param_tol = diff_tolerance(params, new_params)
                    print("DEBUG: param_tol =", param_tol)
                    params = new_params
                    break
                # Reduce step-size
                step_size *= 0.5

            # Obtain new score and check convergence
            new_score = self.compute_score(params, data, controls)
            score_tol = np.abs(new_score - score)
            score = new_score
            if param_tol < controls["param_tol"] or score_tol < controls["score_tol"]:
                print("DEBUG[optimise_parameters]: Converged! params =", params)
                converged = True
                break

        results = {
            "score": score,
            "num_iters": num_iters,
            "score_tol": score_tol,
            "param_tol": param_tol,
            "converged": converged,
        }
        return params, results

    def compute_score(self, params: Values, data: Data, controls: Controls) -> float:
        """
        Computes the score of the objective function for the current parameter
        estimates, averaged over the observed value(s).

        Note: If the objective function requires regularisation of the parameters,
        then the overall score should be modified here. Use controls to specify
        the regularisation weights.

        Input:
            - params (tuple of float or vector): The current values of the parameters.
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - score (float): The mean objective function score.
        """
        return mean_value(data.weights, self.compute_scores(params, data.variate))

    def compute_scores(self, params: Values, variate: Vector) -> Vector:
        """
        Computes the score(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        Input:
            - params (tuple of float or vector): The current values of the parameters.
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - scores (vector): The objective function score(s).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_update(self, params: Values, data: Data, controls: Controls) -> Values:
        """
        Computes a parameter update vector in the direction of improving
        the objective function, evaluated at the current parameter estimates
        and averaged over the observed value(s).

        Note: If the objective function requires regularisation of the parameters,
        then the overall update should be modified here. Use controls to specify
        the regularisation weights.

        Input:
            - params (tuple of float or vector): The current values of the parameters.
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - delta_params (tuple of float or vector): The updates for the parameters.
        """
        raise NotImplementedError


###############################################################################
# Class decorators:


# Decorator for easily overriding the default values of fitting controls
def fitting_controls(
    **controls: Controls,
) -> Callable[[Type[Optimiser]], Type[Optimiser]]:
    """
    Modifies the default values of the convergence controls for the
    parameter estimation algorithm.

    Input:
        - controls (dict): The overriden controls and their new default values.
            See Optimiser.default_controls().

    Returns:
        - decorator (method): A decorator of an optimiser class.
    """

    def decorator(klass: Type[Optimiser]) -> Type[Optimiser]:
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
# Specialised data fitting classes:


class GradientOptimiser(Optimiser):
    """
    Estimates the parameters of the implemented objective function
    via iterative gradient or modified-gradient optimisation.

    The parameter values are stored in a Parameterised instance.

    Note: If second derivatives are not computed then compute_neg_hessian()
    must return a length-zero value, and only gradient optimisation will be used.
    In such a case, step-size conntrol should be applied.

    If second derivatives are computed then Newton-Raphson optimisation will be
    used instead, and step-size control is typically not required.
    """

    def compute_update(self, params: Values, data: Data, controls: Controls) -> Values:
        grad = mean_values(data.weights, self.compute_gradients(params, data.variate))
        n_hess = self.compute_neg_hessian(params, data.variate)
        if len(n_hess) == 0:
            return tuple(grad)
        n_hess = np.array([mean_values(data.weights, r) for r in n_hess])
        return tuple(solve(n_hess, grad))

    @abstractmethod
    def compute_gradients(self, params: Values, variate: Vector) -> Values:
        """
        Computes the gradient(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        Note: If any (first) derivative is independent of the observed data
        and a function only of scalar-valued parameters, then a scalar
        derivative value may be returned instead of the usual vector value.

        Input:
            - params (tuple of float or vector): The current estimates of
                the parameters.
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - grad_params (tuple of float or vector): The derivatives of the
                objective function with respect to the parameters.
        """
        raise NotImplementedError

    def compute_neg_hessian(self, params: Values, variate: Vector) -> Values2d:
        """
        Computes the negative of the Hessian matrix of second derivatives
        of the objective function for the current parameter estimates,
        evaluated at the observed value(s).

        Note: If any second derivative is independent of the observed data
        and a function only of scalar-valued parameters, then a scalar
        derivative value may be returned instead of the usual vector value.

        Input:
            - params (tuple of float or vector): The current estimates of
                the parameters.
            - variate (vector): The observed value(s) of the valiate.

        Returns:
            - Sigma (matrix-like of float or vector): The negative second
                derivatives of the objective function with respect to the
                parameters.
        """
        # By default, do not compute second derivatives
        return tuple()
