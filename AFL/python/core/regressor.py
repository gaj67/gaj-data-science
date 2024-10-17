"""
This module defines the base classes for algorithms to conditionlly fit
observed data, i.e. estimate parameters, using covariates.

It is assumed that some parameterised objective function will be optimised. 
"""

from abc import abstractmethod
from typing import Optional, Type, Callable

import numpy as np
from numpy.linalg import solve

from .data_types import (
    VectorLike,
    MatrixLike,
    Values,
    to_vector,
    to_matrix,
    mean_value,
    mean_values,
    values_to_matrix,
)

from .distribution import Parameterised
from .optimiser import Optimiser, Data, Controls, Results, GradientOptimiser


###############################################################################
# Base class for regression:

# NOTE: Don't make Regressable subclass ABC otherwise overriding fit() method
# via @add_regressor() doesn't remove the abstraction.


class Regressable:
    """
    Provides an interface for estimating parameters from observed
    variate and covariate data.

    Assumes the underlying implementation subclasses Parameterised,
    and that the first parameter specifies the vector of regression
    weights.
    """

    @abstractmethod
    def fit(
        self,
        variate: VectorLike,
        covariates: MatrixLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - covariates (matrix-like): The value(s) of the covariate(s).
            - weights (vector-like, optional): The weight(s) of the data.
            - controls (dict): The user-specified controls.
                See Optimiser.default_controls().

        Returns:
            - results (dict): The summary output. See Optimiser.optimise_parameters().
        """
        raise NotImplementedError

    def to_data(
        self,
        variate: VectorLike,
        covariates: MatrixLike,
        weights: Optional[VectorLike] = None,
    ) -> Data:
        """
        Bundles the observational data into standard format.

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - covariates (matrix-like): The value(s) of the covariate(s).
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
        if isinstance(self, Parameterised):  # Should be true!
            phi = self.parameters()[0]
            n_cols = len(phi) if len(phi) > 0 else -1
        else:
            n_cols = -1
        m_covariates = to_matrix(covariates, n_rows, n_cols)

        return Data(v_data, v_weights, m_covariates)


###############################################################################
# Class decorators:


# Decorator for easily adding an optimiser implementation
def add_regressor(
    fitter_class: Type[Optimiser],
) -> Callable[[Type[Regressable]], Type[Regressable]]:
    """
    Implements the fit() method to wrap the distribution
    with an instance of the specified optimiser.

    Input:
        - fitter_class (class): The class of an optimiser implementation.

    Returns:
        - decorator (method): A decorator of a fittable and parameterised class.

    """

    def decorator(klass: Type[Regressable]) -> Type[Regressable]:

        if not (issubclass(klass, Parameterised) and issubclass(klass, Regressable)):
            raise ValueError("Class must be Parameterised & Regressable!")

        def fit(
            self,  # Parameterised & Regressable
            variate: VectorLike,
            weights: Optional[VectorLike] = None,
            **controls: Controls,
        ) -> Results:
            data = self.to_data(variate, weights)
            return fitter_class(self).fit(data, **controls)

        klass.fit = fit
        klass.fit.__doc__ = Regressable.fit.__doc__
        return klass

    return decorator


###############################################################################
# Abstract implementation of regression:


class GradientRegressor(GradientOptimiser):
    """
    Estimates the parameters of a supplied conditional distribution
    via iterative optimisation of an objective function.

    NOTE: The high-level computation of the score and update is with
    respect to the regression parameters and the independent parameters.

    However, the low-level computation of scores and derivatives is with
    respect to the dependent parameter and the independent parameters.
    """

    def compute_score(self, params: Values, data: Data, controls: Controls) -> float:
        phi, *psi = params
        eta = data.covariates @ phi
        alt_params = eta, *psi
        scores = self.compute_scores(alt_params, data.variate)
        return mean_value(data.weights, scores)

    def compute_update(self, params: Values, data: Data, controls: Controls) -> Values:
        phi, *psi = params
        eta = data.covariates @ phi
        alt_params = eta, *psi
        g_eta, *g_psi = self.compute_gradients(alt_params, data.variate)
        # Map back from eta to phi, i.e. compute <dL/dphi> = <Z dL/deta>
        tot_weights = np.sum(data.weights)
        g_phi = (data.weights * g_eta) @ data.covariates / tot_weights
        if len(g_psi) > 0:
            g_psi = mean_values(data.weights, g_psi)
        n_hess = self.compute_neg_hessian(alt_params, data.variate)
        if len(n_hess) == 0:
            # No second derivatives, just use gradient
            return g_phi, *g_psi

        # Expected value of negative Hessian of log-likelihood gives:
        # [ Var[Y_eta],        Cov[Y_eta, Y_psi] ]
        # [ Cov[Y_psi, Y_eta], Var[Y_psi]        ]

        # Map back from eta to phi
        n_rows = len(data.variate)
        # Row 0 gives -d/deta [dL/deta dL/dpsi] = [Var[Y_eta], Cov[Y_eta, Y_psi]]
        m0 = values_to_matrix(n_hess[0], n_rows)
        n_cols = m0.shape[1]
        # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
        v_phi = (
            sum(
                data.weights[k]
                * m0[k, 0]
                * np.outer(data.covariates[k, :], data.covariates[k, :])
                for k in range(n_rows)
            )
            / tot_weights
        )
        if n_cols > 1:
            # Compute covariance matrix, i.e. cov := <Cov[Y_psi, Y_phi]> = <Cov[Y_psi, Y_eta] Z^T>
            cov = (
                sum(
                    data.weights[k] * np.outer(m0[k, 1:], data.covariates[k, :])
                    for k in range(n_rows)
                )
                / tot_weights
            )
            # Compute variance matrix of Y_psi, i.e. v_psi := <Var[Y_psi]>
            v_psi = np.array(
                [
                    (data.weights @ values_to_matrix(r[1:], n_rows)) / tot_weights
                    for r in n_hess[1:]
                ]
            )

        # The task now is to solve the matrix equation:
        #   [v_phi cov^T] * [d_phi] = [g_phi]
        #   [cov   v_psi]   [d_psi]   [g_psi]
        # using block matrix inversion. An answer (for v_phi invertible) is:
        #   d_psi = S^-1 * (g_psi - cov * v_phi^-1 * g_phi),
        #   d_phi = v_phi^-1 * (g_phi - cov^T * d_psi),
        # where S is the Schur complement:
        #   S = v_psi - cov * v_phi^-1 * cov^T.

        # Inversion step 1: d_phi' = v_phi^-1 * g_phi
        d_phi = solve(v_phi, g_phi)
        if n_cols == 1:
            # No independent parameters - just return d_phi
            return (d_phi,)
        # Inversion step 2: S = v_psi - cov * v_phi^-1 * cov^T
        schur = v_psi - cov @ solve(v_phi, cov.T)
        # Inversion step 3: d_psi = S^-1 * (g_psi - cov * d_phi')
        d_psi = solve(schur, g_psi - cov @ d_phi)
        # Inversion step 4: d_phi = d_phi' - v_phi^-1 * cov^T * d_psi
        d_phi -= solve(v_phi, d_psi @ cov)
        return d_phi, *d_psi
