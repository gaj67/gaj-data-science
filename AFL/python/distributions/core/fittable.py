"""
This modules defines classes for estimating parameters from data
(without covariates) using an optimiser.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.linalg import solve

from .data_types import (
    Values,
    Values2d,
    Vector,
    VectorLike,
    mean_value,
    mean_values,
)

from .optimiser import (
    Controllable,
    Optimisable,
    Optimiser,
    Results,
    Controls,
    Data,
    to_data,
)


###############################################################################
# Interface classes for gradient optimisation:


class Scorable(ABC):
    """
    An interface for computing the score(s) of
    an objective function for given data.
    """

    @abstractmethod
    def compute_scores(self, variate: Vector) -> Vector:
        """
        Computes the score(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        Input:
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - scores (vector): The objective function score(s).
        """
        raise NotImplementedError


class Differentiable(ABC):
    """
    An interface for computing the gradient, and optionally the Hessian,
    of an objective function.

    Note: If second derivatives are not computed then compute_neg_hessian()
    must return a length-zero value, and only the gradient will be used.
    """

    @abstractmethod
    def compute_gradients(self, variate: Vector) -> Values:
        """
        Computes the gradient(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        Note: If any (first) derivative is independent of the observed data
        and a function only of scalar-valued parameters, then a scalar
        derivative value may be returned instead of the usual vector value.

        Input:
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - grad_params (tuple of float or vector): The derivatives of the
                objective function with respect to the parameters.
        """
        raise NotImplementedError

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        """
        Computes the negative of the Hessian matrix of second derivatives
        of the objective function for the current parameter estimates,
        evaluated at the observed value(s).

        Note: If any second derivative is independent of the observed data
        and a function only of scalar-valued parameters, then a scalar
        derivative value may be returned instead of the usual vector value.

        Input:
            - variate (vector): The observed value(s) of the valiate.

        Returns:
            - n_hess (matrix-like of float or vector): The negative second
                derivatives of the objective function with respect to the
                parameters.
        """
        # By default, do not compute second derivatives
        return tuple()


###############################################################################
# Implementation classes for optimisation:


class StandardOptimisable(Optimisable, Scorable, Differentiable):
    """
    Implements a standard optimisable objective function using gradients.

    No covariate information is used.
    """

    def compute_score(self, data: Data, controls: Controls) -> float:
        return mean_value(data.weights, self.compute_scores(data.variate))

    def compute_update(self, data: Data, controls: Controls) -> Values:
        grad = mean_values(data.weights, self.compute_gradients(data.variate))
        n_hess = self.compute_neg_hessian(data.variate)
        if len(n_hess) == 0:
            return tuple(grad)
        n_hess = np.array([mean_values(data.weights, r) for r in n_hess])
        return tuple(solve(n_hess, grad))


class Fittable(StandardOptimisable, Controllable):
    """
    Interface for estimating parameter values from data.
    """

    def fit(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls
    ) -> Results:
        """
        Estimates optimal parameter values from the data.

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - weights (vector-like, optional): The weight(s) of the data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - res (dict): The summary of the estimation algorithm.
        """
        _data = to_data(variate, weights)
        _controls = self.get_controls(**controls)
        _optimiser = Optimiser(self)
        return _optimiser.fit(_data, _controls)
