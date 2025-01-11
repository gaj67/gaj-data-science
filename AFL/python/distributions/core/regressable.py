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

from .parameterised import RegressionParameters, UNSPECIFIED_VECTOR
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


class RegressionOptimisable(RegressionParameters, Optimisable):
    """
    An implementation for parameter estimation from observed
    variate and covariate data.

    Assumes the underlying implementation is a regression model
    with its first parameter being a vector of regression weights,
    and any remaining parameters being the independent parameters
    of an underlying link model.

    Here we futher assume that the link model is differentiable
    annd optimisable.
    """

    # ------------------------------
    # RegressionParameters interface

    def __init__(self, num_links: int, link_model: GradientOptimisable):
        """
        Initialises the regression parameters.
        
        Input:
            - num_links (int): The required number
                of regression parameters.
            - link_model (optimisable): The optimisable link model.
        """
        RegressionParameters.__init__(self, num_links, link_model)

    def link_model(self) -> GradientOptimisable:
        """
        Obtains the underlying link model.

        Returns:
            - link_model (optimisable): The link model.
        """
        return super().link_model()

    # -------------------------------
    # RegressionOptimisable interface

    def apply_regression(self, covariates: MatrixLike) -> Values:
        """
        Computes the regression function(s) and then pushes the
        resulting link parameter(s) into the underlying distribution.

        Input:
            - covariates (matrix-like): The covariate value(s).
        """
        phis = self.regression_model().get_parameters()
        num_links = len(phis)
        if num_links == 0:
            return
        vec_len = len(phis[0])
        for phi in phis:
            if len(phi) == 0:
                raise ValueError("Uninitialised regression parameters!")
            if len(phi) != vec_len:
                raise ValueError("Incorrectly sized regression parameters!")
        z = to_matrix(covariates, n_cols=vec_len)
        etas = [as_value(z @ phi) for phi in phis]
        psi = self.link_model().get_parameters()[num_links:]
        self.link_model().set_parameters(*etas, *psi)

    # ---------------------
    # Optimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        ind, values = self.link_model().compute_estimates(data.variate)
        if len(values) == 0 or not np.any(ind):
            # No point estimates - use mean estimate and zero weights
            params = self.link_model().compute_estimate(data, controls)
            psi = params[self.num_links():]
            vec_len = data.covariates.shape[1]
            phis = [np.zeros(vec_len) for _ in range(self.num_links())]
        else:
            # Average point estimates of independent parameters
            w = data.weights[ind]
            psi = mean_values(w, values[self.num_links():])
            # Link parameter(s) data estimates
            etas = values[0:self.num_links()]
            print("DEBUG[RegressionOptimisable.compute_estimate]: etas=", etas)
            z = data.covariates[ind, :]
            print("DEBUG[RegressionOptimisable.compute_estimate]: z=", z)
            a_mat = sum_vmm(w, z, z)
            # Invert linear regression(s): Z @ phi = eta
            phis = [solve(a_mat, (w * eta) @ z) for eta in etas]
        print("DEBUG[RegressionOptimisable.compute_estimate]: phis=", phis)
        print("DEBUG[RegressionOptimisable.compute_estimate]: psi=", psi)
        return (*phis, *psi)

    def compute_score(self, data: Data, controls: Controls) -> float:
        self.apply_regression(data.covariates)
        scores = self.link_model().compute_scores(data.variate)
        return mean_value(data.weights, scores)

    def compute_update(self, data: Data, controls: Controls) -> Values:
        self.apply_regression(data.covariates)
        grads = self.link_model().compute_gradients(data.variate)
        if len(grads) == 0:
            # No update
            return tuple()

        # Compute means of point gradients of the independent parameters
        n_links = self.num_links()
        g_psi = mean_values(data.weights, grads[n_links:])
        print("DEBUG[RegressionOptimisable.compute_update]: g_psi=", g_psi)

        # Map back from eta to phi, i.e. compute <dL/dphi> = <Z dL/deta>
        g_etas = grads[0:n_links]
        w = data.weights / np.sum(data.weights)
        g_phis = [(w * g_eta) @ data.covariates for g_eta in g_etas]
        print("DEBUG[RegressionOptimisable.compute_update]: g_phis=", g_phis)

        neg_hess = self.link_model().compute_neg_hessian(data.variate)
        print("DEBUG[RegressionOptimisable.compute_update]: n_hess=", neg_hess)
        if len(neg_hess) == 0:
            # No second derivatives, just use gradient
            return (*g_phis, *g_psi)

        # Compute neg_hess[phi, psi]^-1 * grads[phi, psi]
        g_phi = UNSPECIFIED_VECTOR if n_links == 0 else np.concatenate(g_phis)
        d_phi, d_psi = self._compute_modified_gradients(neg_hess, g_phi, g_psi, data)
        print("DEBUG[RegressionOptimisable.compute_update]: d_phi=", d_phi)
        print("DEBUG[RegressionOptimisable.compute_update]: d_psi=", d_psi)
        d_phis = [] if n_links == 0 else np.split(d_phi, n_links)
        return (*d_phis, *d_psi)

    def _compute_modified_gradients(self, neg_hess: Values2d, g_phi: Vector, g_psi: Vector, data: Data) -> Tuple[Vector, Vector]:
        # Expected value of negative Hessian of link log-likelihood gives:
        #  [ Var[Y_eta]         Cov[Y_eta, Y_psi] ]
        #  [ Cov[Y_psi, Y_eta]  Var[Y_psi]        ]
        # Under regression eta = Z^T phi, this is mapped to:
        #  [v_phi cov^T] = [<Z v_eta Z^T>  <Z cov_ep>]
        #  [cov   v_psi]   [<cov_pe Z^T>   <v_psi>   ]

        v_phi, cov, v_psi = self._compute_blocks(neg_hess, data)
        print(
            "DEBUG[RegressionOptimisable.compute_update]: v_phi=",
            v_phi,
            "v_psi=",
            v_psi,
            "cov=",
            cov,
        )

        # The general task now is to solve the matrix equation:
        #   [v_phi cov^T] * [d_phi] = [g_phi]
        #   [cov   v_psi]   [d_psi]   [g_psi]
        # using block matrix inversion. An answer (for v_phi invertible) is:
        #   d_psi = S^-1 * (g_psi - cov * v_phi^-1 * g_phi),
        #   d_phi = v_phi^-1 * (g_phi - cov^T * d_psi),
        # where S is the Schur complement:
        #   S = v_psi - cov * v_phi^-1 * cov^T.

        # Check for no link parameters:
        if v_phi is None:
            # Inversion step 2: S = v_psi
            # Inversion step 3: d_psi = S^-1 * g_psi
            d_psi = solve(v_psi, g_psi)
            return UNSPECIFIED_VECTOR, d_psi

        # Inversion step 1: d_phi' = v_phi^-1 * g_phi
        d_phi = solve(v_phi, g_phi)

        # Check for no independent parameters:
        if v_psi is None:
            # Only link parameters
            return d_phi, UNSPECIFIED_VECTOR

        # Inversion step 2: S = v_psi - cov * v_phi^-1 * cov^T
        schur = v_psi - cov @ solve(v_phi, cov.T)
        # Inversion step 3: d_psi = S^-1 * (g_psi - cov * d_phi')
        d_psi = solve(schur, g_psi - cov @ d_phi)
        # Inversion step 4: d_phi = d_phi' - v_phi^-1 * cov^T * d_psi
        d_phi -= solve(v_phi, cov.T @ d_psi)
        return d_phi, d_psi

    def _compute_blocks(
        self, neg_hess: Values2d, data: Data
    ) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[Matrix]]:
        n_links = self.num_links()
        n_params = len(neg_hess)
        n_indep = n_params - n_links

        n_rows = len(data.variate)
        w = data.weights / np.sum(data.weights)
        z = data.covariates

        # Check for no link parameters:
        if n_links == 0:
            # Compute variance matrix of Y_psi, i.e. v_psi := <Var[Y_psi]>
            v_psi = np.array([(w @ values_to_matrix(r, n_rows)) for r in neg_hess])
            return None, None, v_psi

        # Check for no independent parameters:
        if n_indep == 0:
            # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
            v_phis = [[sum_vmm(w * v, z, z) for v in r] for r in neg_hess]
            v_phi = np.vstack([np.hstack(r) for r in v_phis])
            return v_phi, None, None

        # Both link parameters and independent parameters:
        # Compute variance matrix of Y_psi, i.e. v_psi := <Var[Y_psi]>
        v_psi = np.array([(w @ values_to_matrix(r[n_links:], n_rows)) for r in neg_hess[n_links:]])
        # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
        v_phis = [[sum_vmm(w * v, z, z) for v in r[0:n_links]] for r in neg_hess[0:n_links]]
        v_phi = np.vstack([np.hstack(r) for r in v_phis])
        # Compute covariance matrix, i.e. cov := <Cov[Y_psi, Y_phi]> = <Cov[Y_psi, Y_eta] Z^T>
        covs = [sum_vmm(w, values_to_matrix(r[n_links:], n_rows), z) for r in neg_hess[0:n_links]]
        cov = np.hstack(covs)
        return v_phi, cov, v_psi


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
