"""
This modules defines classes used to implement the computations underlying parameter estimation.
"""

from abc import abstractmethod
from typing import Type, Tuple

import numpy as np
from numpy.linalg import solve

from .parameterised import Parameterised, TransformParameterised
from .controllable import Controls

from .data_types import (
    Values,
    Values2d,
    Vector,
    Data,
    mean_value,
    mean_values,
    mult_rmat_vec,
    mult_rmat_rmat,
)


###############################################################################
# Standardised interface for objective function optimisation:


class Optimisable(Parameterised):
    """
    A standardised interface for estimating parameters from data by
    optimising an objective function.
    """

    def estimate_parameters(
        self, data: Data, controls: Controls  # pylint: disable=unused-argument
    ) -> Values:
        """
        Computes initial estimates of the parameters from observed data.

        If a closed-form solution of the optimisation does not exist,
        then the initial estimates will typically be approximate values,
        usually computed via the method of moments or another heuristic.

        Inputs:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - params (tuple of scalar or vector): The estimated value(s)
                of the parameters.
        """
        values = self.compute_estimates(data.variate)
        if len(values) == 0:
            return self.default_parameters()
        ind = np.ones(len(data.variate), dtype=bool)
        for est in values:
            ind &= ~np.isnan(est)
        if not np.any(ind):
            return self.default_parameters()
        return tuple(mean_values(data.weights, values, use_only=ind))

    def compute_estimates(
        self, variate: Vector  # pylint: disable=unused-argument
    ) -> Values:
        """
        If possible, computes point estimates of the parameters for every
        observation of interest.

        Note: If no point estimates are feasible, then return an empty tuple.
        Consider overriding estimate_parameters() instead.
        If any given point estimate is infeasible, then set it to NaN.
        These non-estimates will then be ignored.

        Input:
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - params (tuple of scalar or vector): The estimated value(s)
                of the parameter(s).
        """
        # By default, do not compute point estimates
        return tuple()

    def evaluate_score(
        self, data: Data, controls: Controls  # pylint: disable=unused-argument
    ) -> float:
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
        return mean_value(data.weights, self.compute_scores(data.variate))

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

    @abstractmethod
    def estimate_update(self, data: Data, controls: Controls) -> Values:
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
# Specialised interface for target-based parameter optimisation:


class TargetOptimisable(Optimisable):
    """
    A standardised interface for estimating parameters from data by optimising
    the discrepancy between model predictions and empirical target values.
    """

    @abstractmethod
    def target_scores(self, data: Data, controls: Controls) -> Vector:
        """
        Computes target scores from an empirical model of the observed data.

        Input:
            - data (tuple of data): The observational data.
            - controls (dict): Additional user-supplied information.

        Returns:
            - scores (vector): The target score for every observation.
        """
        raise NotImplementedError


###############################################################################
# Extends iterative optimisation to enable gradient-based optimisation:


class GradientOptimisable(Optimisable):
    """
    Interface for an optimisable objective function using
    derivatives with respect to the parameters.

    Note: By default, any covariate information remains unused.
    """

    # ---------------------
    # Optimisable interface

    def estimate_update(self, data: Data, controls: Controls) -> Values:
        grads = self.compute_gradients(data.variate)
        if len(grads) == 0:
            # No update
            return tuple()
        grad = mean_values(data.weights, grads)
        hess = self.compute_hessian(data.variate)
        if len(hess) == 0:
            return tuple(grad)
        hess = np.array([mean_values(data.weights, r) for r in hess])
        return tuple(solve(hess, -grad))

    # -----------------------------
    # GradientOptimisable interface

    def compute_gradients(
        self, variate: Vector  # pylint: disable=unused-argument
    ) -> Values:
        """
        Computes the gradient(s) of the objective function for the
        current parameter estimates, evaluated at the observed value(s).

        Note: If any (first) derivative is independent of the observed data
        and a function only of scalar-valued parameters, then a scalar
        derivative value may be returned instead of the usual vector value.

        Note: It is assumed that gradient ascent is used to maximise the
        objective function. If minimisation is required via gradient-descent,
        then return the negative of the gradient.

        Input:
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - grads (tuple of scalar or vector): The derivatives,
                across all data-points, of the objective function
                with respect to each parameter.
        """
        # By default, do not compute first derivatives
        return tuple()

    def compute_hessian(
        self, variate: Vector  # pylint: disable=unused-argument
    ) -> Values2d:
        """
        Computes the Hessian matrix of second derivatives
        of the objective function for the current parameter estimates,
        evaluated at the observed value(s).

        Note: If any second derivative is independent of the observed data
        and a function only of scalar-valued parameters, then a scalar
        value may be returned instead of the usual vector value.

        Input:
            - variate (vector): The observed value(s) of the valiate.

        Returns:
            - hess (matrix-like of scalar or vector): The second
                derivatives of the objective function with respect to the
                parameters.
        """
        # By default, do not compute second derivatives
        return tuple()


# Special decorator for turning off second derivative calculations:
def no_hessian(klass: Type[GradientOptimisable]) -> Type[GradientOptimisable]:
    """
    Turns off modified-gradient calculations by forcing the return
    of an empty Hessian matrix.
    """
    _compute_hessian = klass.compute_hessian

    def compute_hessian(
        self, variate: Vector  # pylint: disable=unused-argument
    ) -> Values2d:
        # Do not compute second derivatives
        return tuple()

    klass.compute_hessian = compute_hessian
    klass.compute_hessian.__doc__ = _compute_hessian.__doc__
    return klass


###############################################################################
# Special class for gradient optimisation in a transformed space:


class TransformOptimisable(TransformParameterised, GradientOptimisable):
    """
    Partial implementattion of an optimisable objective function using
    scores and derivatives computed in a transformed space.

    Assumes the existence of an underlying instance to hold the actual
    parameters, and to compute the objective function score and derivatives
    in the original space.

    By default, no covariate information is used.
    """

    # ------------------------------
    # TransformOptimisable interface

    def __init__(self, obj: GradientOptimisable):
        """
        Initialises the transormation with the underlying parameters.

        Input:
            - obj (optimisable): The instance containing the underlying
                parameters of the untransformed parameter space.
        """
        TransformParameterised.__init__(self, obj)

    def underlying(self) -> GradientOptimisable:
        """
        Obtains the underlying optimisable model.

        Returns:
            - obj (optimisable): The underlying instance.
        """
        return super().underlying()

    @abstractmethod
    def compute_jacobian(self) -> Values2d:
        """
        Computes the Jacobian matrix J = dx^T/dy of the inverse
        transformation x = f^{-1}(y), where x^T denotes the row-vector
        of standard parameters and y denotes the column-vector
        of transformed parameters, such that dx = J^T dy.
        Thus, for some other function g(x), we also obtain
        dg/dy = J dg/dx .

        Returns:
            - jac (matrix-like of scalar or vector): The Jacobian matrix
                of the inverse transformation.
        """
        raise NotImplementedError

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        alt_values = self.compute_estimates(data.variate)
        if len(alt_values) == 0:
            # Transform mean estimate
            std_params = self.underlying().estimate_parameters(data, controls)
            return self.apply_transform(*std_params)
        # Take mean of tranformed estimates
        return tuple(mean_values(data.weights, alt_values))

    def compute_estimates(self, variate: Vector) -> Tuple[Vector, Values]:
        """
        If possible, computes point estimates of the parameters for every
        observation of interest.

        Note: If no point estimates are feasible (marked by an empty tuple),
        or if any single point estimate is infeasible (marked as NaN),
        then the entire point estimation process is abandoned.

        Input:
            - variate (vector): The observed value(s) of the variate.

        Returns:
            - params (tuple of scalar or vector): The estimated value(s)
                of the parameter(s).
        """
        std_values = self.underlying().compute_estimates(variate)
        if len(std_values) == 0:
            return tuple()
        for est in std_values:
            if np.any(np.isnan(est)):
                return tuple()
        alt_values = self.apply_transform(*std_values)
        return alt_values

    def compute_scores(self, variate: Vector) -> Vector:
        return self.underlying().compute_scores(variate)

    def compute_gradients(self, variate: Vector) -> Values:
        grads = self.underlying().compute_gradients(variate)
        if len(grads) == 0:
            return tuple()
        jac = self.compute_jacobian()
        return mult_rmat_vec(jac, grads)

    def compute_hessian(self, variate: Vector) -> Values2d:
        hess = self.underlying().compute_hessian(variate)
        if len(hess) == 0:
            return tuple()
        jac = self.compute_jacobian()
        # This is an approximation which assumes that the
        # expectation of any score gradient is identically zero.
        return mult_rmat_rmat(mult_rmat_rmat(jac, hess), jac)
