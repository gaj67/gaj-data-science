"""
This modules defines classes for estimating parameters from 
variate and covariate data using an optimiser.
"""

from typing import Optional, Tuple
from abc import abstractmethod

import numpy as np
from numpy.linalg import solve

from .data_types import (
    Values,
    Values2d,
    VectorLike,
    Matrix,
    MatrixLike,
    mean_value,
    mean_values,
    to_matrix,
    values_to_matrix,
    as_value,
)
from .parameterised import Parameterised
from .optimiser import (
    Controls,
    Data,
    Results,
    Optimisable,
    Optimiser,
    Controllable,
    to_data,
)
from .fittable import GradientOptimisable


###############################################################################
# Classes for gradient optimisation with covariates using linear regression:


class RegressionOptimisable(GradientOptimisable):
    """
    An interface for parameter estimation from observed variate and
    covariate data.

    Assumes the underlying implementation is a regression model
    with its first parameter being a vector of regression weights,
    i.e. a RegressionParameters instance.

    Also assumes that the regression model is connected to a
    supplied link model, and that this link model is differentiable.
    """

    @abstractmethod
    def underlying(self) -> Linkable:
        """
        Obtains the underlying link model.

        Returns:
            - link (linkable): The link model.
        """
        raise NotImplementedError

    # ---------------------
    # Optimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        _, *psi = self.default_parameters()
        phi = np.zeros(data.covariates.shape[1])
        return (phi, *psi)

    def compute_score(self, data: Data, controls: Controls) -> float:
        self._invert_regression(data.covariates)
        scores = self.underlying().compute_scores(data.variate)
        return mean_value(data.weights, scores)

    def compute_update(self, data: Data, controls: Controls) -> Values:
        self._invert_regression(data.covariates)
        p = self.underlying()
        g_eta, *g_psi = p.compute_gradients(data.variate)
        # Map back from eta to phi, i.e. compute <dL/dphi> = <Z dL/deta>
        tot_weights = np.sum(data.weights)
        g_phi = (data.weights * g_eta) @ data.covariates / tot_weights
        if len(g_psi) > 0:
            g_psi = mean_values(data.weights, g_psi)

        n_hess = p.compute_neg_hessian(data.variate)
        if len(n_hess) == 0:
            # No second derivatives, just use gradient
            return g_phi, *g_psi

        # Expected value of negative Hessian of link log-likelihood gives:
        #  [ Var[Y_eta]         Cov[Y_eta, Y_psi] ]
        #  [ Cov[Y_psi, Y_eta]  Var[Y_psi]        ]
        # Under regression eta = Z^T phi, this is mapped to:
        #  [v_phi cov^T] = [Z v_eta Z^T  Z cov_ep]
        #  [cov   v_psi]   [cov_pe Z^T   v_psi   ]

        v_phi, cov, v_psi = self._compute_blocks(n_hess, data)

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
        if len(g_psi) == 0:
            # No independent parameters, so no d_psi
            return (d_phi,)
        # Inversion step 2: S = v_psi - cov * v_phi^-1 * cov^T
        schur = v_psi - cov @ solve(v_phi, cov.T)
        # Inversion step 3: d_psi = S^-1 * (g_psi - cov * d_phi')
        d_psi = solve(schur, g_psi - cov @ d_phi)
        # Inversion step 4: d_phi = d_phi' - v_phi^-1 * cov^T * d_psi
        d_phi -= solve(v_phi, d_psi @ cov)
        return d_phi, *d_psi

    # Regression interface

    def _compute_blocks(
        self, n_hess: Values2d, data: Data
    ) -> Tuple[Matrix, Optional[Matrix], Optional[Matrix]]:
        # Map back from eta to phi
        n_rows = len(data.variate)
        # Note: Row 0 gives -d/deta [dL/deta dL/dpsi] = [Var[Y_eta], Cov[Y_eta, Y_psi]]
        m0 = values_to_matrix(n_hess[0], n_rows)
        n_cols = m0.shape[1]
        # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
        tot_weights = np.sum(data.weights)
        v_phi = (
            sum(
                data.weights[k]
                * m0[k, 0]
                * np.outer(data.covariates[k, :], data.covariates[k, :])
                for k in range(n_rows)
            )
            / tot_weights
        )
        if n_cols == 1:
            # No independent parameters, so no cov or v_psi
            return v_phi, None, None
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
        return v_phi, cov, v_psi

    def _invert_regression(self, covariates: MatrixLike) -> Values:
        """
        Computes the regression function and then inverts the
        link parameter into the underlying distribution.

        Input:
            - covariates (matrix-like): The covariate value(s).
        """
        phi = self.parameters()[0]
        if len(phi) == 0:
            raise ValueError("Uninitialised regression parameters!")
        covs = to_matrix(covariates, n_cols=len(phi))
        eta = as_value(covs @ phi)
        _, *psi = self.underlying().parameters()
        self.underlying().set_parameters(eta, *psi)


#################################################################
# The main regression class:


class Regressable(Optimisable, Controllable):
    """
    An interface for parameter estimation from observed variate and
    covariate data.
    """

    def fit(
        self,
        variate: VectorLike,
        covariates: MatrixLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates optimal parameter values from the data.

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - covariates (matrix-like): The value(s) of the covariate(s).
            - weights (vector-like, optional): The weight(s) of the data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - res (dict): The summary of the estimation algorithm.
        """
        _data = to_data(variate, weights, covariates)
        _controls = self.get_controls(**controls)
        _optimiser = Optimiser(self)
        return _optimiser.fit(_data, _controls)
