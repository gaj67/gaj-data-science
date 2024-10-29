"""
This module defines the base classes for algorithms to conditionlly fit
observed data, i.e. estimate parameters, using covariates.

It is assumed that some parameterised objective function will be optimised. 
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
from numpy.linalg import solve

from .data_types import (
    Value,
    Values,
    Vector,
    VectorLike,
    MatrixLike,
    to_vector,
    to_matrix,
    as_value,
    mean_value,
    mean_values,
    values_to_matrix,
)

from .parameterised import Parameterised
from .controllable import Controls
from .estimator import Estimator, Data, Results, Differentiable


# Indicates that thee regression weights have not yet been specified
UNSPECIFIED_REGRESSION = np.array([])


###############################################################################
# Base class for the regression parameterisation:


class Regression(Parameterised):
    """
    Indicates the use of a parameterised regression function of the covariates, Z.
    """

    def __init__(self, reg_params: Vector, *indep_params: Values):
        """
        Initialises the regression parameters.

        Use the UNSPECIFIED_REGRESSION constant to indicate that the regression
        parameters are unknown, and need to be set or estimated.

        Input:
            - reg_params (vector): The value(s) of the regression parameter(s).
            - indep_params (tuple of float, optional): The value(s) of the
                independent parameter(s), if any.
        """
        super().__init__(reg_params, *indep_params)

    def regression_parameters(self) -> Vector:
        """
        Obtains the regression parameters.

        Returns:
            - reg_params (vector): The value(s) of the regression parameter(s).
        """
        return self.parameters()[0]

    def independent_parameters(self) -> Values:
        """
        Obtains the independent parameters.

        Returns:
            - indep_params (tuple of float): The value(s) of the independent
                parameter(s), if any.
        """
        return self.parameters()[1:]

    def apply_regression(self, covariates: MatrixLike) -> Value:
        """
        Computes the regression function to obtain the link parameter.

        Input:
            - covariates (matrix-like): The covariate value(s).

        Returns:
            - link_param (float or vector): The value(s) of the link parameeter.
        """
        phi = self.regression_parameters()
        if len(phi) == 0:
            raise ValueError("Uninitialised regression parameters!")
        covs = to_matrix(covariates, n_cols=len(phi))
        eta = as_value(covs @ phi)
        return eta

    @abstractmethod
    def invert_link(self, *link_params: Values) -> Values:
        """
        Inverts the link function to map the link parameter and any independent
        parameters into the corresponding distributional parameters.

        Input:
            - link_params (tuple of float or vector): The value(s) of the link
                parameter, along with the value(s) of the independent parameter(s),
                if any.

        Returns:
            - inv_params (tuple of float or vector): The values(s) of the
                distributional parameter(s).
        """
        raise NotImplementedError


###############################################################################
# Base class for estimating the regression parameters:


class RegressionEstimator(Estimator):
    """
    Estimates the regression parameters and independent parameters, either via a
    closed-form solution or via iterative optimisation of an objective function.
    """

    def estimate_parameters(
        self,
        data: Data,
        **controls: Controls,
    ) -> Results:
        if not isinstance(self, Regression):
            raise ValueError("Instance is not a regression!")

        return super().estimate_parameters(data, **controls)

    def compute_score(self, params: Values, data: Data, controls: Controls) -> float:
        phi, *psi = params
        eta = data.covariates @ phi
        alt_params = eta, *psi
        scores = self.compute_scores(alt_params, data.variate)
        print("DEBUG[compute_score]: score=", mean_value(data.weights, scores))
        return mean_value(data.weights, scores)

    def compute_update(self, params: Values, data: Data, controls: Controls) -> Values:
        if not isinstance(self, Differentiable):
            raise NotImplementedError("Non-gradient update is not implemented!")

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
            print("DEBUG[compute_update]: g_phi=", g_phi, "g_psi=", g_psi)
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


###############################################################################
# Simple data  fitter:


class Fittable(RegressionEstimator):
    """
    Provides an interface for estimating parameters from observed
    variate and covariate data.

    Assumes the underlying implementation is a regression, and that
    the first parameter specifies the vector of regression weights.
    """

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
        data = self.to_data(variate, covariates, weights)
        return self.estimate_parameters(data, **controls)

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
        if isinstance(self, Regression):  # Should be true!
            phi = self.regression_parameters()
            n_cols = len(phi) if len(phi) > 0 else -1
        else:
            n_cols = -1
        m_covariates = to_matrix(covariates, n_rows, n_cols)

        return Data(v_data, v_weights, m_covariates)
