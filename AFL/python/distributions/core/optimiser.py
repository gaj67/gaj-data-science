"""
This modules defines classes for parameter estimation.
Typically this is achieved by optimising an objective function.
"""

from typing import TypeAlias, Tuple, Dict, Any

import numpy as np
from scipy.optimize import minimize, curve_fit

from .data_types import (
    Values,
    Vector,
    Data,
    is_vector,
)
from .optimisable import Optimisable, TargetOptimisable
from .controllable import Controls, Controllable, set_controls


###############################################################################
# Base class for optimisation:


# Encapsulates the output summary of the optimisation algorithm.
Results: TypeAlias = Dict[str, Any]


# Helper method for testing convergence
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


@set_controls(
    # (bool) Permits a controls.get("debug") trace
    debug=False,
    # (bool) Controls whether or not parameters are re-initialised
    initialise=True,
    # (int) The maximum number of iterations allowed
    max_iters=100,
    # (float) The minimum change in score to signal convergence
    score_tol=1e-8,
    # (float) The minimum change in parameter values to signal convergence
    param_tol=1e-6,
    # (float) The parameter update scaling factor (or learning rate)
    step_size=1.0,
    # (float): The line-search step-size reduction factor
    step_decay=0.5,
    # (int) The maximum permitted number of step-size line-searches
    step_iters=10,
)
class Optimiser(Controllable):
    """
    Default implementation of an iterative parameter estimator.
    """

    def __init__(self, obj: Optimisable):
        """
        Encapsulates the instance for which optimal parameters are to be estimated.

        Input:
            - obj (optimisable): A parameter optimisable instance.
        """
        self._obj = obj

    def underlying(self) -> Optimisable:
        """
        Obtains the underlying instance.

        Returns:
            - obj (optimisable): The parameter optimisable instance.
        """
        return self._obj

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
        _controls = self.get_controls(**controls)

        # Initialise the parameter values
        if _controls.get("initialise"):
            obj = self.underlying()
            params = obj.estimate_parameters(data, _controls)
            obj.set_parameters(*params)

        # Iteratively optimise the parameters
        res = self.optimise_parameters(data, _controls)
        return res

    def optimise_parameters(self, data: Data, controls: Controls) -> Results:
        """
        Updates the value(s) of the parameter(s) by maximising the mean score
        of the objective function for the given observations.

        Input:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls.

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
        obj = self.underlying()

        res = {
            "score": obj.evaluate_score(data, controls),
            "num_iters": 0,
            "score_tol": 0.0,
            "param_tol": 0.0,
            "converged": False,
        }
        if controls.get("debug"):
            print("DEBUG: controls =", controls)
            print("DEBUG[0]: score =", res["score"])
            print("DEBUG[0]: parameters =", obj.get_parameters())

        min_score_tol = controls.get("score_tol")
        min_param_tol = controls.get("param_tol")

        for _it in range(controls.get("max_iters")):
            # Obtain update
            d_params = obj.estimate_update(data, controls)
            if controls.get("debug"):
                print(f"DEBUG[{_it}]: update =", d_params)
            if len(d_params) == 0:
                # No update available!
                break

            # Apply line search
            res["num_iters"] += 1
            step_size = controls.get("step_size")
            step_decay = controls.get("step_decay")
            params = obj.get_parameters()
            for _ls in range(controls.get("step_iters")):
                # Apply update
                new_params = tuple(v + step_size * d for v, d in zip(params, d_params))
                if controls.get("debug"):
                    print(f"DEBUG[{_it}:{_ls}]: parameters =", new_params)
                if obj.check_parameters(*new_params):
                    # Tentatively accept update
                    res["param_tol"] = diff_tolerance(params, new_params)
                    obj.set_parameters(*new_params)
                    # Obtain new score and check for improvement
                    new_score = obj.evaluate_score(data, controls)
                    if controls.get("debug"):
                        print(f"DEBUG[{_it}:{_ls}]: score =", new_score)
                    res["score_tol"] = new_score - res["score"]
                    if (
                        new_score >= res["score"]
                        or min_score_tol > 0
                        and np.abs(res["score_tol"]) < min_score_tol
                    ):
                        # Accept parameter update
                        break
                # Reject update and try again
                obj.set_parameters(*params)
                step_size *= step_decay
            else:
                # Could not improve estimate
                raise ValueError("Parameters failed to converge within bounds!")

            # Check convergence
            res["score"] = new_score
            res["converged"] = (
                min_param_tol <= 0 or res["param_tol"] < min_param_tol
            ) and (min_score_tol <= 0 or np.abs(res["score_tol"]) < min_score_tol)
            if res["converged"]:
                break

        # Algorithm has halted
        return res


###############################################################################
# External, nonlinear optimisation:


class NonlinearOptimiser(Optimiser):
    """
    External implementation of a parameter optimiser.
    A numerical minimiser is used by default.
    """

    def __init__(self, obj: Optimisable):
        super().__init__(obj)
        params = obj.get_parameters()
        self._param_lengths = [len(v) if is_vector(v) else -1 for v in params]

    def encode_parameters(self, params: Values) -> Vector:
        """
        Encodes the parameters into a flattened vector.

        Input:
            - params (tuple of float or vector): The value(s) of the parameters.
        Returns:
            - x_params (vector): The flattened vector.
        """
        return np.hstack(params)

    def decode_parameters(self, x_params: Vector) -> Values:
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
        for plen in self._param_lengths:
            if plen < 0:
                # Scalar parameeter
                params.append(x_params[start])
                start += 1
            else:
                # Vector parameter
                end = start + plen
                params.append(x_params[start:end])
                start = end
        return tuple(params)

    def optimise_parameters(self, data: Data, controls: Controls) -> Results:
        """
        Estimates the value(s) of the parameter(s) by minimising the negative mean score
        of the objective function for the given observations.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls.

        Returns:
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - converged (bool): Indicates whether or not the minimisation
                    converged.
        """
        obj = self.underlying()

        def score_and_grad_fn(x_params: Vector) -> Tuple[float, Vector]:
            """
            Computes the score and (pseudo-)gradient for the current estimates.

            Input:
                - x_params (vector): The flattened parameter estimates.

            Returns:
                - score (float): The current score.
                - grad (vector): The flattened parameter gradients.
            """
            params = self.decode_parameters(x_params)
            obj.set_parameters(*params)
            d_params = obj.estimate_update(data, controls)
            if len(d_params) == 0:
                raise ValueError("Could not compute parameter update!")
            score = obj.evaluate_score(data, controls)
            xd_params = self.encode_parameters(d_params)
            return (-score, -xd_params)  # pylint: disable=invalid-unary-operand-type

        # Optimise score
        res = minimize(
            score_and_grad_fn, self.encode_parameters(obj.get_parameters()), jac=True
        )
        return {
            "score": -res.fun,
            "num_iters": res.nit,
            "converged": res.status == 0,
        }


###############################################################################
# External, least-squares optimisation:


@set_controls(
    # (function) Used to transform scores prior to fitting
    score_fn=None,
)
class LeastSquaresOptimiser(NonlinearOptimiser):
    """
    External implementation of a target-score parameter optimiser.
    Least-squares fitting is used by default.
    """

    def __init__(self, obj: Optimisable):
        super().__init__(obj)
        if not isinstance(obj, TargetOptimisable):
            raise ValueError("Required interface: TargetOptimisable")

    def optimise_parameters(self, data: Data, controls: Controls) -> Results:
        """
        Estimates the value(s) of the parameter(s) by minimising the mean
        squared difference between the theoretical and empirical models.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): The user-specified controls.

        Returns:
            - results (dict): The summary output, including:
                - score (float): The final mean score of the data.
                - num_iters (int): The number of iterations performed.
                - converged (bool): Indicates whether or not the minimisation
                    converged.
        """
        obj = self.underlying()

        # Define score transformation function
        if controls.get("score_fn", None) is not None:
            score_fn = controls.get("score_fn")
        else:
            # Use identity function
            def score_fn(score: float) -> float:
                return score

        def model_fn(x_data: Vector, *x_params: Tuple) -> Vector:
            """
            Computes the data-point scores using the current parameter estimates.

            Input:
                - x_data (vector): The observed values of the variate.
                - x_params (vector): The flattened parameter estimates.

            Returns:
                - scores (vector): The current scores.
            """
            params = self.decode_parameters(x_params)
            obj.set_parameters(*params)
            return score_fn(obj.compute_scores(x_data))

        # Estimate empirical model
        y_data = score_fn(obj.target_scores(data, controls))

        # Optimise score fit
        x_params, *info = curve_fit(
            model_fn,
            data.variate,
            y_data,
            sigma=1.0 / np.sqrt(data.weights + 1e-30),
            p0=self.encode_parameters(obj.get_parameters()),
            full_output=True,
        )
        # Compute score
        obj.set_parameters(*self.decode_parameters(x_params))
        score = obj.evaluate_score(data, controls)
        m_data = score_fn(obj.compute_scores(data.variate))
        mse = np.mean(data.weights * (y_data - m_data) ** 2)
        # Return status
        return {
            "score": score,
            "mse": mse,
            "num_iters": info[1]["nfev"],
            "converged": True,
        }
