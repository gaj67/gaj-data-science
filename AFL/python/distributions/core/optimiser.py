"""
This modules defines classes for parameter estimation.
Typically this is achieved by optimising an objective function.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, Dict, Any, NamedTuple, Optional, Tuple, Callable, Type

import numpy as np
from scipy.optimize import minimize

from .data_types import (
    Values,
    Vector,
    VectorLike,
    Matrix,
    MatrixLike,
    is_vector,
    to_vector,
    to_matrix,
)
from .parameterised import Parameterised


###############################################################################
# Data-types and methods for optimisation:

# Encapsulates the performance summary of the parameter estimation algorithm.
Results: TypeAlias = Dict[str, Any]

# Encapsulates any and all settings necessary to control the algorithm.
# See Controllable.default_controls().
Controls: TypeAlias = Dict[str, Any]


class Data(NamedTuple):
    """
    Provides a wrapper for observed data.
    """

    variate: Vector
    weights: Vector
    covariates: Optional[Matrix] = None


def to_data(
    variate: VectorLike,
    weights: Optional[VectorLike] = None,
    covariates: Optional[MatrixLike] = None,
) -> Data:
    """
    Bundles the observational data into standard format.

    Inputs:
        - variate (vector-like): The value(s) of the variate.
        - weights (vector-like, optional): The weight(s) of the data.
        - covariates (matrix-like, optional): The value(s) of the covariate(s).

    Returns:
        - data (tuple of data): The bundled data.
    """
    # Allow for single or multiple observations, with or without weights
    v_data = to_vector(variate)
    n_rows = len(v_data)
    if weights is None:
        v_weights = np.ones(n_rows)
    else:
        v_weights = to_vector(weights, n_rows)
    if covariates is None:
        return Data(v_data, v_weights)
    m_covariates = to_matrix(covariates, n_rows)
    return Data(v_data, v_weights, m_covariates)


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
# Specialised interface for an objective function:


class Optimisable(Parameterised):
    """
    An interface for estimating parameters from data by
    optimising an objective function.
    """

    @abstractmethod
    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        """
        Computes initial estimates of the parameters from observed data.

        If a closed-form solution of the optimisation does not exist,
        then the initial estimates will typically be approximate values,
        usually computed via the method of moments.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - params (tuple of scalar or vector): The estimated value(s)
                of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_score(self, data: Data, controls: Controls) -> float:
        """
        Computes the score of the objective function for the current parameter
        estimates, averaged over the observed value(s).

        Note: If the objective function requires regularisation of the parameters,
        then the overall score should be modified here. Use controls to specify
        the regularisation weights.

        Input:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - score (float): The mean objective function score.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_update(self, data: Data, controls: Controls) -> Values:
        """
        Computes a parameter update vector in the direction of improving
        the objective function, evaluated at the current parameter estimates
        and averaged over the observed value(s).

        Note: If the objective function requires regularisation of the parameters,
        then the overall update should be modified here. Use controls to specify
        the regularisation weights.

        Note: If no update is available, then return an empty result.

        Input:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - delta_params (tuple of float or vector): The updates for the parameters.
        """
        raise NotImplementedError


###############################################################################
# Base class for optimisation:


class Optimiser:
    """
    Implementation of an iterative parameter estimator.
    """

    def __init__(self, inst: Optimisable):
        """
        Encapsulates the parameters to be estimated.

        Input:
            - inst (optimisable): The holder of the parameters.
        """
        self._inst = inst

    def underlying(self) -> Optimisable:
        """
        Obtains the underlying instance.

        Returns:
            - inst (optimisable): The holder of the parameters.
        """
        return self._inst

    def fit(
        self,
        data: Data,
        controls: Controls,
    ) -> Results:
        """
        Estimates optimal parameter values from data.

        Input:
            - data (Data): The bundled data.
            - controls (Controls): The bundled controls.

        Returns:
            - res (Results): The summary results.
        """
        # Initialise the parameter values
        p = self.underlying()
        if controls.get("init", True):
            params = p.compute_estimate(data, controls)
            p.set_parameters(*params)

        # Iteratively optimise the parameters
        if controls.get("use_external", False):
            return self._external_optimiser(data, controls)
        return self._internal_optimiser(data, controls)

    def _internal_optimiser(self, data: Data, controls: Controls) -> Results:
        """
        Iteratively estimates the values of the parameters by optimising
        the mean score of the objective function for the given observations.

        Input:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - score_tol (float): The final score-change tolerance.
                - param_tol (float): The final parameter-change tolerance.
                - converged (bool): Indicates whether or not parameter
                    and score convergence was achieved.
        """
        # Get score of current parameters
        p = self.underlying()

        score = p.compute_score(data, controls)
        num_iters = 0
        converged = False

        res = {
            "score": score,
            "num_iters": num_iters,
            "score_tol": 0.0,
            "param_tol": 0.0,
            "converged": converged,
        }

        min_score_tol = controls.get("score_tol", 0.0)
        min_param_tol = controls.get("param_tol", 0.0)

        while num_iters < controls.get("max_iters", 0):
            # Obtain update
            d_params = p.compute_update(data, controls)
            print("DEBUG[optimiser]: d_params=", d_params)
            if len(d_params) == 0:
                # No update available!
                break

            # Apply line search
            num_iters += 1
            step_size = controls.get("step_size", 1.0)
            params = p.get_parameters()
            print("DEBUG[optimiser]: params=", params)
            for _ in range(controls.get("step_iters", 5)):
                # Apply update
                new_params = tuple(v + step_size * d for v, d in zip(params, d_params))
                print("DEBUG[optimiser]: new_params=", new_params)
                print("DEBUG[optimiser]: valid=", p.check_parameters(*new_params))
                if p.check_parameters(*new_params):
                    # Tentatively accept update
                    res["param_tol"] = diff_tolerance(params, new_params)
                    p.set_parameters(*new_params)
                    # Obtain new score and check for improvement
                    new_score = p.compute_score(data, controls)
                    print("DEBUG[optimiser]: score=", score, "new_score=", new_score)
                    if new_score >= score:
                        # Accept parameter update
                        break
                # Reject update and try again
                p.set_parameters(*params)
                step_size *= 0.5
            else:
                # Could not improve estimate
                raise ValueError("Parameters failed to converge within bounds!")

            # Check convergence
            res["score_tol"] = new_score - score
            score = new_score
            converged = (min_param_tol <= 0 or res["param_tol"] <= min_param_tol) and (
                min_score_tol <= 0 or res["score_tol"] <= min_score_tol
            )
            if converged:
                break

        res["score"] = score
        res["num_iters"] = num_iters
        res["converged"] = converged

        return res

    def _external_optimiser(self, data: Data, controls: Controls) -> Results:
        """
        Iteratively estimates the values of the parameters by minimising the
        negative mean score of the objective function for the given observations.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - converged (bool): Indicates whether or not the minimisation
                    converged.
        """
        p = self.underlying()
        params = p.get_parameters()
        lengths = [len(v) if is_vector(v) else -1 for v in params]

        def encode(*params: Values) -> Vector:
            """
            Encodes the parameters into a flattened vector.

            Input:
                - params (tuple of float or vector): The value(s) of
                    the parameters.
            Returns:
                - x_params (vector): The flattened vector.
            """
            return np.hstack(params)

        def decode(x_params: Vector) -> Values:
            """
            Decodes the flattened vector into a tuple of parameters.

            Input:
                - x_params (vector): The flattened vector.

            Returns:
                - params (tuple of float or vector): The value(s) of
                    the parameters.
            """
            params = []
            start = 0
            for l in lengths:
                if l < 0:
                    params.append(x_params[start])
                    start += 1
                else:
                    end = start + l
                    params.append(x_params[start:end])
                    start = end
            return tuple(params)

        def score_and_grad_fn(x_params: Vector) -> Tuple[float, Vector]:
            """
            Computes tthe score and gradient foor the current estimates.

            Input:
                - x_params (vector): The flattened parameter estimates.

            Returns:
                - score (float): The current scoore.
                - grad (vector): The flattened parameter gradients.
            """
            params = decode(x_params)
            p.set_parameters(*params)
            score = p.compute_score(data, controls)
            d_params = p.compute_update(data, controls)
            if len(d_params) == 0:
                raise ValueError("Could not compute update!")
            return -score, -encode(*d_params)

        # Optimise score
        x0 = encode(*params)
        res = minimize(score_and_grad_fn, x0, jac=True)
        return {
            "score": -res.fun,
            "num_iters": res.nit,
            "converged": res.status == 0,
        }


###############################################################################
# Class for controlling the optimiser:


class Controllable(ABC):
    """
    Encapsulates the typical controls and their default values.
    """

    @staticmethod
    def default_controls() -> Controls:
        """
        Provides default settings for controlling the algorithm.

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

    def get_controls(
        self,
        **controls: Controls,
    ) -> Controls:
        """
        Permits the default control values to be dynamically overridden.

        Input:
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output. See optimise_parameters().
        """
        # Obtain specified controls
        _controls = self.default_controls()
        _controls.update(controls)
        return _controls


###############################################################################
# Decorator for controlling the optimiser:


# Decorator for easily overriding the default values of controls
def set_controls(
    **controls: Controls,
) -> Callable[[Type[Controllable]], Type[Controllable]]:
    """
    Statically modifies the default values of the algorithm's controls.

    Input:
        - controls (dict): The overriden controls and their new default values.
            See Controllable.default_controls().

    Returns:
        - decorator (method): A decorator of a controllable class.
    """

    def decorator(klass: Type[Controllable]) -> Type[Controllable]:
        default_controls_fn = klass.default_controls

        @staticmethod
        def default_controls() -> Controls:
            _controls = default_controls_fn()
            _controls.update(controls)
            return _controls

        klass.default_controls = default_controls
        klass.default_controls.__doc__ = default_controls_fn.__doc__
        return klass

    return decorator
