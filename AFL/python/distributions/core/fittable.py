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
    mult_rmat_vec,
    mult_rmat_rmat,
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
# Base class for gradient optimisation:


class GradientOptimisable(Optimisable):
    """
    Interface for an optimisable objective function using gradients.

    By default, no covariate information is used.
    """

    # ---------------------
    # Optimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        values = self.compute_estimates(data.variate)
        if len(values) == 0:
            raise NotImplementedError("Override compute_estimate(s)!")
        return tuple(mean_values(data.weights, values))

    def compute_score(self, data: Data, controls: Controls) -> float:
        return mean_value(data.weights, self.compute_scores(data.variate))

    def compute_update(self, data: Data, controls: Controls) -> Values:
        grads = self.compute_gradients(data.variate)
        if len(grads) == 0:
            # No update
            return tuple()
        grad = mean_values(data.weights, grads)
        n_hess = self.compute_neg_hessian(data.variate)
        if len(n_hess) == 0:
            return tuple(grad)
        n_hess = np.array([mean_values(data.weights, r) for r in n_hess])
        return tuple(solve(n_hess, grad))

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimates(self, variate: Vector) -> Values:
        """
        Computes point estimates of the parameters for every
        observation.

        Note: If point estimates are not feasible then return an
        empty result and override compute_estimate().

        Input:
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - params (tuple of scalar or vector): The estimated value(s)
                of the parameter(s).
        """
        # By default, do not compute point estimates
        return tuple()

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
        # By default, do not compute first derivatives
        return tuple()

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
# Special class for gradient optimisation in a transformed space:


class TransformOptimisable(GradientOptimisable):
    """
    Interface for an optimisable objective function using scores and derivatives
    computed in a transformed space.

    Assumes the existence of an underlying instance to hold the actual
    parameters, and to compute the objective function score and derivatives
    in the original space.

    By default, no covariate information is used.
    """

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        std_params = self.underlying().default_parameters()
        return self.apply_transform(*std_params)

    def parameters(self) -> Values:
        std_params = self.underlying().parameters()
        return self.apply_transform(*std_params)

    def set_parameters(self, *params: Values):
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")
        std_params = self.invert_transform(*params)
        self.underlying().set_parameters(*std_params)

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        std_params = self.underlying().compute_estimate(data, controls)
        return self.apply_transform(*std_params)

    def compute_scores(self, variate: Vector) -> Vector:
        return self.underlying().compute_scores(variate)

    def compute_gradients(self, variate: Vector) -> Values:
        grads = self.underlying().compute_gradients(variate)
        if len(grads) == 0:
            return tuple()
        jac = self.compute_jacobian()
        return mult_rmat_vec(jac, grads)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        n_hess = self.underlying().compute_neg_hessian(variate)
        if len(n_hess) == 0:
            return tuple()
        jac = self.compute_jacobian()
        # This is an approximation which assumes that
        # the expectation of any score gradient is zero.
        return mult_rmat_rmat(mult_rmat_rmat(jac, n_hess), jac)

    # ------------------------------
    # TransformOptimisable interface

    @abstractmethod
    def underlying(self) -> GradientOptimisable:
        """
        Obtains the underlying model.

        Returns:
            - inst (optimisable): The underlying instance.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_transform(self, *std_params: Values) -> Values:
        """
        Transforms the standard parameterisation into the
        alternative parameterisation.

        Input:
            - std_params (tuple of float or vector): The value(s)
                of the standard parameter(s).

        Returns:
            - alt_params (tuple of float or vector): The value(s)
                of the alternative parameter(s).
        """
        raise NotImplementedError

    @abstractmethod
    def invert_transform(self, *alt_params: Values) -> Values:
        """
        Inversely transforms the alternative parameterisation into
        the standard parameterisation.

        Input:
            - alt_params (tuple of float or vector): The value(s)
                of the alternative parameter(s).

        Returns:
            - std_params (tuple of float or vector): The value(s)
                of the standard parameter(s).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_jacobian(self) -> Values2d:
        """
        Computes the Jacobian matrix of the inverse transformation,
        i.e. the derivatives of the standard parameters (column-wise)
        with respect to the alternative parameters (row-wise).

        Returns:
            - jac (matrix-like of scalar or vector): The Jacobian matrix
                of the inverse link transformation.
        """
        raise NotImplementedError


###############################################################################
# Main class for optimisation:


class Fittable(Optimisable, Controllable):
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
