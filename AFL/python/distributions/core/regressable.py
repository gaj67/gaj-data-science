"""
This modules defines classes for estimating parameters from 
variate and covariate data using an optimiser.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.linalg import solve

from .data_types import (
    Values,
    Values2d,
    Vector,
    VectorLike,
    Matrix,
    MatrixLike,
    Data,
    to_data,
    mean_value,
    mean_values,
    to_matrix,
    values_to_matrix,
    as_value,
    sum_vmm,
)
from .controllable import Controls, Controllable
from .optimisable import Optimisable, GradientOptimisable
from .optimiser import Results, Optimiser
from .parameterised import Parameterised, VectorParameters, UNSPECIFIED_VECTOR


###############################################################################
# Interface for regression parameters


# The default value of the link parameter
DEFAULT_LINK = 0.0


class RegressionParameters(Parameterised):
    """
    An implementation for accessing regression parameters and
    associated independent parameters.

    It is assumed that there are two underlying models.
    The regression model contains the regression parameters.
    The link model contains the link parameters followed by
    any independent parameters.

    Each scalar link parameter, eta, and corresponding
    vector regression parameter, phi, are assumed to be
    directly related by a regression function of the
    covariates, Z, namely: eta = f(Z, phi).
    Although this function is not used explicitly here, we do
    make use of the assumption that f(Z, 0) = 0.
    """

    # ------------------------------
    # RegressionParameters interface

    def __init__(self, num_links: int, link_model: Parameterised):
        """
        Initialises the regression model parameters.

        Input:
            - num_links (int): The required number
                of regression parameters.
            - link_model (parameterised): The underlying link model.
        """
        self._num_links = num_links
        self._regression_parameters = VectorParameters(num_links)
        self._link_model = link_model

    def regression_model(self) -> Parameterised:
        """
        Obtains the underlying regression model.

        Returns:
            - regression (parameterised): The regression model.
        """
        return self._regression_parameters

    def link_model(self) -> Parameterised:
        """
        Obtains the underlying link model.

        Returns:
            - link_model (parameterised): The link model.
        """
        return self._link_model

    def num_links(self) -> int:
        """
        Obtains the number of regressable link parameters of
        the underlying link model.

        Returns:
            - num_links (int): The number of link parameters.
        """
        return self._num_links

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return self.link_model().num_parameters()

    def default_parameters(self) -> Values:
        phis = self.regression_model().default_parameters()
        psi = self.link_model().default_parameters()[self.num_links() :]
        return (*phis, *psi)

    def get_parameters(self) -> Values:
        phis = self.regression_model().get_parameters()
        psi = self.link_model().get_parameters()[self.num_links() :]
        return (*phis, *psi)

    def set_parameters(self, *params: Values):
        if not self.check_parameters(*params):
            raise ValueError("Invalid parameters!")
        phis = params[0 : self._num_links]
        psi = params[self._num_links :]
        self.regression_model().set_parameters(*phis)
        etas = [DEFAULT_LINK] * self.num_links()
        self.link_model().set_parameters(*etas, *psi)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != self.num_parameters():
            return False
        phis = params[0 : self.num_links()]
        psi = params[self.num_links() :]
        if not self.regression_model().check_parameters(*phis):
            return False
        etas = [DEFAULT_LINK] * self.num_links()
        return self.link_model().check_parameters(*etas, *psi)


###############################################################################
# Classes for gradient optimisation with covariates using linear regression:


class RegressionOptimisable(RegressionParameters, Optimisable):
    """
    An implementation for parameter estimation from observed
    variate and covariate data.

    Assumes the underlying implementation is a regression model
    with its first n parameters being vectors of regression weights,
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
        z_mat = to_matrix(covariates, n_cols=vec_len)
        etas = [as_value(z_mat @ phi) for phi in phis]
        psi = self.link_model().get_parameters()[num_links:]
        self.link_model().set_parameters(*etas, *psi)

    # ---------------------
    # Optimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        values = self.link_model().compute_estimates(data.variate)
        if len(values) > 0:
            ind = np.ones(len(data.variate), dtype=bool)
            for est in values:
                ind &= ~np.isnan(est)
            if not np.any(ind):
                values = tuple()
        if len(values) == 0:
            # No point estimates - use mean estimate and zero weights
            params = self.link_model().estimate_parameters(data, controls)
            psi = params[self.num_links() :]
            vec_len = data.covariates.shape[1]
            phis = [np.zeros(vec_len) for _ in range(self.num_links())]
        else:
            # Average point estimates of independent parameters
            weights = data.weights[ind]
            psi = mean_values(weights, values[self.num_links() :])
            # Link parameter(s) data estimates
            etas = values[0 : self.num_links()]
            z_mat = data.covariates[ind, :]
            a_mat = sum_vmm(weights, z_mat, z_mat)
            # Invert linear regression(s): Z @ phi = eta
            phis = [solve(a_mat, (weights * eta[ind]) @ z_mat) for eta in etas]
        return (*phis, *psi)

    def evaluate_score(self, data: Data, controls: Controls) -> float:
        self.apply_regression(data.covariates)
        scores = self.link_model().compute_scores(data.variate)
        return mean_value(data.weights, scores)

    def compute_scores(self, variate: Vector) -> Vector:
        # There are no given covariates!
        # Hence, we cannot use least-squares or nonlinear optimisation.
        raise NotImplementedError

    def estimate_update(self, data: Data, controls: Controls) -> Values:
        self.apply_regression(data.covariates)
        grads = self.link_model().compute_gradients(data.variate)
        if len(grads) == 0:
            # No update
            return tuple()

        # Compute means of point gradients of the independent parameters
        n_links = self.num_links()
        g_psi = mean_values(data.weights, grads[n_links:])

        # Map back from eta to phi, i.e. compute <dL/dphi> = <Z dL/deta>
        g_etas = grads[0:n_links]
        weights = data.weights / np.sum(data.weights)
        g_phis = [(weights * g_eta) @ data.covariates for g_eta in g_etas]

        hess = self.link_model().compute_hessian(data.variate)
        if len(hess) == 0:
            # No second derivatives, just use gradient
            return (*g_phis, *g_psi)

        # Compute -hess[phi, psi]^-1 * grads[phi, psi]
        g_phi = UNSPECIFIED_VECTOR if n_links == 0 else np.concatenate(g_phis)
        d_phi, d_psi = self._compute_modified_gradients(hess, g_phi, g_psi, data)
        d_phis = [] if n_links == 0 else np.split(d_phi, n_links)
        return (*d_phis, *d_psi)

    def _compute_modified_gradients(
        self, hess: Values2d, g_phi: Vector, g_psi: Vector, data: Data
    ) -> Tuple[Vector, Vector]:
        # Expected value of negative Hessian of link log-likelihood gives:
        #  [ Var[Y_eta]         Cov[Y_eta, Y_psi] ]
        #  [ Cov[Y_psi, Y_eta]  Var[Y_psi]        ]
        # Under regression eta = Z^T phi, this is mapped to:
        #  [v_phi cov^T] = [<Z v_eta Z^T>  <Z cov_ep>]
        #  [cov   v_psi]   [<cov_pe Z^T>   <v_psi>   ]

        v_phi, cov, v_psi = self._compute_blocks(hess, data)

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
        self, hess: Values2d, data: Data
    ) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[Matrix]]:
        """
        Computes the block components of the expected negated-Hessian matrix.
        """
        n_links = self.num_links()
        n_params = len(hess)
        n_indep = n_params - n_links

        n_rows = len(data.variate)
        # Quick hack - negate the weights vector instead of the Hessian Values2d object
        weights = -data.weights / np.sum(data.weights)
        z_mat = data.covariates

        # Check for no link parameters:
        if n_links == 0:
            # Compute variance matrix of Y_psi, i.e. v_psi := <Var[Y_psi]>
            v_psi = np.array(
                [(weights @ values_to_matrix(r_i, n_rows)) for r_i in hess]
            )
            return None, None, v_psi

        # Check for no independent parameters:
        if n_indep == 0:
            # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
            v_phis = [
                [sum_vmm(weights * v_ij, z_mat, z_mat) for v_ij in r_i] for r_i in hess
            ]
            v_phi = np.vstack([np.hstack(r_i) for r_i in v_phis])
            return v_phi, None, None

        # Both link parameters and independent parameters:
        # Compute variance matrix of Y_psi, i.e. v_psi := <Var[Y_psi]>
        v_psi = np.array(
            [
                (weights @ values_to_matrix(r_i[n_links:], n_rows))
                for r_i in hess[n_links:]
            ]
        )
        # Compute variance matrix of Y_phi, i.e. v_phi := <Var[Y_phi]> = <Z Var[Y_eta] Z^T>
        v_phis = [
            [sum_vmm(weights * v_ij, z_mat, z_mat) for v_ij in r_i[0:n_links]]
            for r_i in hess[0:n_links]
        ]
        v_phi = np.vstack([np.hstack(r_i) for r_i in v_phis])
        # Compute covariance matrix, i.e. cov := <Cov[Y_psi, Y_phi]> = <Cov[Y_psi, Y_eta] Z^T>
        covs = [
            sum_vmm(weights, values_to_matrix(r_i[n_links:], n_rows), z_mat)
            for r_i in hess[0:n_links]
        ]
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
