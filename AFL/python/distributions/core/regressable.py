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
    Vector,
    VectorLike,
    Matrix,
    MatrixLike,
    mean_value,
    mean_values,
    to_matrix,
    values_to_matrix,
    as_value,
)
from .optimiser import (
    Controls,
    Data,
    Results,
    Optimisable,
    Optimiser,
    Controllable,
    to_data,
)

from .parameterised import RegressionParameterised
from .fittable import GradientOptimisable


###############################################################################
# Helper methods for linear regression:


def sum_vmm(vec: Vector, mat1: Matrix, mat2: Matrix) -> Matrix:
    """
    Computes sum_i {v_i X_i Y_i^T}.

    Input:
        - vec (vector): The weight vector, [v_i].
        - mat1 (matrix): The first row matrix, [X_i^T].
        - mat2 (matrix): The second row matrix, [Y_i^T].

    Returns:
        - res (matrix): The computed result.
    """
    return sum(v * np.outer(x, y) for v, x, y in zip(vec, mat1, mat2))


###############################################################################
# Classes for gradient optimisation with covariates using linear regression:


class RegressionOptimisable(RegressionParameterised, Optimisable):
    """
    An interface for parameter estimation from observed variate and
    covariate data.

    Assumes the underlying implementation is a regression model
    with its first parameter being a vector of regression weights,
    and any remaining parameters being the independent parameters
    of an underlying link model.

    Here we futher assume that the link model is differentiable
    annd optimisable.
    """

    # ---------------------------------
    # RegressionParameterised interface

    @abstractmethod
    def link(self) -> GradientOptimisable:
        """
        Obtains the underlying link model.

        Returns:
            - inst (optimisable): The link model.
        """
        raise NotImplementedError

    # ---------------------
    # Optimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        ind, values = self.link().compute_estimates(data.variate)
        if len(values) == 0 or not np.any(ind):
            # No point estimates - use mean estimate and zero weights
            _, *psi = self.link().compute_estimate(data, controls)
            phi = np.zeros(data.covariates.shape[1])
            print(
                "DEBUG[RegressionOptimisable.compute_estimate]: phi=", phi, "psi=", psi
            )
        else:
            eta, *psi = values
            print("DEBUG[RegressionOptimisable.compute_estimate]: eta=", eta)
            # Solve linear regression: Z @ phi = eta
            w = data.weights[ind]
            z = data.covariates[ind, :]
            print("DEBUG[RegressionOptimisable.compute_estimate]: z=", z)
            a_mat = sum_vmm(w, z, z)
            b_vec = (w * eta) @ z
            phi = solve(a_mat, b_vec)
            # Average point estimates of independent parameters
            psi = mean_values(w, psi)
            print(
                "DEBUG[RegressionOptimisable.compute_estimate]: phi=", phi, "psi=", psi
            )
        return (phi, *psi)

    def compute_score(self, data: Data, controls: Controls) -> float:
        self._invert_regression(data.covariates)
        scores = self.link().compute_scores(data.variate)
        return mean_value(data.weights, scores)

    def compute_update(self, data: Data, controls: Controls) -> Values:
        self._invert_regression(data.covariates)
        grads = self.link().compute_gradients(data.variate)
        if len(grads) == 0:
            # No update
            return tuple()
        g_eta, *g_psi = grads

        # Map back from eta to phi, i.e. compute <dL/dphi> = <Z dL/deta>
        tot_weights = np.sum(data.weights)
        g_phi = (data.weights * g_eta) @ data.covariates / tot_weights
        g_psi = mean_values(data.weights, g_psi)
        print(
            "DEBUG[RegressionOptimisable.compute_update]: g_phi=",
            g_phi,
            "g_psi=",
            g_psi,
        )

        n_hess = self.link().compute_neg_hessian(data.variate)
        print("DEBUG[RegressionOptimisable.compute_update]: n_hess=", n_hess)
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
        print(
            "DEBUG[RegressionOptimisable.compute_update]: v_phi=",
            v_phi,
            "v_psi=",
            v_psi,
            "cov=",
            cov,
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
        if len(g_psi) == 0:
            # No independent parameters, so no d_psi
            print("DEBUG[RegressionOptimisable.compute_update]: d_phi=", d_phi)
            return (d_phi,)
        # Inversion step 2: S = v_psi - cov * v_phi^-1 * cov^T
        schur = v_psi - cov @ solve(v_phi, cov.T)
        # Inversion step 3: d_psi = S^-1 * (g_psi - cov * d_phi')
        d_psi = solve(schur, g_psi - cov @ d_phi)
        # Inversion step 4: d_phi = d_phi' - v_phi^-1 * cov^T * d_psi
        d_phi -= solve(v_phi, d_psi @ cov)
        print(
            "DEBUG[RegressionOptimisable.compute_update]: d_phi=",
            d_phi,
            "d_psi=",
            d_psi,
        )
        return d_phi, *d_psi

    # -------------------------------
    # RegressionOptimisable interface

    def _compute_blocks(
        self, n_hess: Values2d, data: Data
    ) -> Tuple[Matrix, Optional[Matrix], Optional[Matrix]]:
        # Map back from eta to phi
        n_rows = len(data.variate)
        # Note: Row 0 gives -d/deta [dL/deta dL/dpsi] = [Var[Y_eta], Cov[Y_eta, Y_psi]]
        m0 = values_to_matrix(n_hess[0], n_rows)
        n_cols = m0.shape[1]
        # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
        w = data.weights / np.sum(data.weights)
        z = data.covariates
        v_phi = sum_vmm(w * m0[:, 0], z, z)
        if n_cols == 1:
            # No independent parameters, so no cov or v_psi
            return v_phi, None, None
        # Compute covariance matrix, i.e. cov := <Cov[Y_psi, Y_phi]> = <Cov[Y_psi, Y_eta] Z^T>
        cov = sum_vmm(w, m0[:, 1:], z)
        # Compute variance matrix of Y_psi, i.e. v_psi := <Var[Y_psi]>
        v_psi = np.array([(w @ values_to_matrix(r[1:], n_rows)) for r in n_hess[1:]])
        return v_phi, cov, v_psi

    def _invert_regression(self, covariates: MatrixLike) -> Values:
        """
        Computes the regression function and then pushes the
        resulting link parameter into the underlying distribution.

        Input:
            - covariates (matrix-like): The covariate value(s).
        """
        phi, *psi = self.get_parameters()
        if len(phi) == 0:
            raise ValueError("Uninitialised regression parameters!")
        z = to_matrix(covariates, n_cols=len(phi))
        eta = as_value(z @ phi)
        self.link().set_parameters(eta, *psi)


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
