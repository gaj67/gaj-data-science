"""
This module defines the base classes for algorithms to fit observed data,
i.e. estimate parameters.
"""

from typing import Tuple
from abc import ABC, abstractmethod
from typing import TypeAlias, Dict, Any, NamedTuple, Optional

import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize

from .data_types import (
    Values,
    Values2d,
    Vector,
    Matrix,
    VectorLike,
    is_vector,
    to_vector,
    mean_value,
    mean_values,
)

from .parameterised import Parameterised
from .controllable import Controls, Controllable


###############################################################################
# Useful data-types and methods:

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
# Specialised interfacees for parameter updates:


class Differentiable(ABC):
    """
    Permits the estimator to use gradient or modified gradient updates.

    Note: If second derivatives are not computed then compute_neg_hessian()
    must return a length-zero value, and only gradient optimisation will be used.
    In such a case, step-size conntrol should be applied if necessary.

    If second derivatives are computed then Newton-Raphson optimisation will be
    used instead, and step-size control is typically not required.
    """

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


###############################################################################
# Abstract data fitting classes:


class Estimator(Controllable):
    """
    Estimates the parameters either via a closed-form solution
    or via iterative optimisation of an objective function.
    """

    def estimate_parameters(
        self,
        data: Data,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - results (dict): The summary output.
                See optimise_parameters().
        """
        # Get all controls
        controls = self.get_controls(**controls)

        # Initialise parameter values.
        if not isinstance(self, Parameterised):
            raise ValueError("Instance is not parameterised!")
        if controls["init"]:
            params = self.initialise_parameters(data, controls)
            self.set_parameters(*params)

        # Iteratively optimise the parameters
        if controls.get("use_external", False):
            return self._external_optimiser(data, controls)
        return self._internal_optimiser(data, controls)

    @abstractmethod
    def initialise_parameters(self, data: Data, controls: Controls) -> Values:
        """
        Estimates initial values of the parameters from observed data.

        If a closed-form solution of the optimisation does not exist,
        then the initial estimates will typically be approximate values,
        usually computed via the method of moments.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - params (tuple of float or vector): The estimated values of the
                parameters.
        """
        raise NotImplementedError

    def _internal_optimiser(self, data: Data, controls: Controls) -> Results:
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
        params = self.parameters()
        score = self.compute_score(params, data, controls)
        score_tol = 0.0
        param_tol = 0.0
        num_iters = 0
        converged = False

        while num_iters < controls["max_iters"]:
            # Obtain update
            d_params = self.compute_update(params, data, controls)

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
                if self.is_valid_parameters(*new_params):
                    param_tol = diff_tolerance(params, new_params)
                    params = new_params
                    self.set_parameters(*params)
                    break
                # Reduce step-size
                step_size *= 0.5

            # Obtain new score and check convergence
            new_score = self.compute_score(params, data, controls)
            score_tol = np.abs(new_score - score)
            score = new_score
            converged = (
                controls["param_tol"] <= 0 or param_tol < controls["param_tol"]
            ) and (controls["score_tol"] <= 0 or score_tol < controls["score_tol"])
            if converged:
                break

        return {
            "score": score,
            "num_iters": num_iters,
            "score_tol": score_tol,
            "param_tol": param_tol,
            "converged": converged,
        }

    def _external_optimiser(self, data: Data, controls: Controls) -> Results:
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
        params = self.parameters()
        lengths = [len(p) if is_vector(p) else -1 for p in params]
        print("DEBUG: lengths=", lengths)

        def encode(*params: Values) -> Vector:
            """
            Encodes the parameters into a flattened vector.

            Input:
                - params (tuple of float or vector): The value(s) of
                    the parameters.
            Returns:
                - vec (vector): The flattened vector.
            """
            return np.hstack(params)

        def decode(vec: Vector) -> Values:
            """
            Decodes the flattened vector into a tuple of parameters.

            Input:
                - vec (vector): The flattened vector.

            Returns:
                - params (tuple of float or vector): The value(s) of
                    the parameters.
            """
            params = []
            start = 0
            for l in lengths:
                if l < 0:
                    params.append(vec[start])
                    start += 1
                else:
                    end = start + l
                    params.append(vec[start:end])
                    start = end
            return tuple(params)

        num_iters = 0

        def score_and_gradients_fn(x: Vector) -> Tuple[float, Vector]:
            nonlocal num_iters
            num_iters += 1
            print("**DEBUG[score_and_gradients_fn]: x=", x)
            params = decode(x)
            print("**DEBUG[score_and_gradients_fn]: params=", params)
            score = self.compute_score(params, data, controls)
            d_params = self.compute_update(params, data, controls)
            print("DEBUG[score_and_gradients_fn]: score=", score, "d_params=", d_params)
            return -score, -encode(*d_params)

        # Optimise score
        print("DEBUG: params=", params)
        x0 = encode(*params)
        print("DEBUG: x0=", x0)
        res = minimize(score_and_gradients_fn, x0, jac=True)
        print("DEBUG: num_iters=", num_iters, "res=", res)
        print("DEBUG: type(res)=", type(res))
        print("DEBUG: dir(res)=", dir(res))
        return {
            "score": -res.fun,
            "num_iters": num_iters,
            "converged": res.status == 0,
        }

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

    @abstractmethod
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
        if not isinstance(self, Differentiable):
            raise NotImplementedError("Non-gradient update is not implemented!")

        grad = mean_values(data.weights, self.compute_gradients(params, data.variate))
        n_hess = self.compute_neg_hessian(params, data.variate)
        if len(n_hess) == 0:
            return tuple(grad)
        n_hess = np.array([mean_values(data.weights, r) for r in n_hess])
        return tuple(solve(n_hess, grad))


###############################################################################
# Simple data fitter:


class Fittable(Estimator):
    """
    Provides an interface for estimating parameters from data.
    """

    def fit(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - weights (vector-like, optional): The weight(s) of the data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - results (dict): The summary output.
        """
        data = self.to_data(variate, weights)
        return self.estimate_parameters(data, **controls)

    def to_data(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
    ) -> Data:
        """
        Bundles the observational data into standard format.

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - weights (vector-like, optional): The weight(s) of the data.

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
        return Data(v_data, v_weights, None)
