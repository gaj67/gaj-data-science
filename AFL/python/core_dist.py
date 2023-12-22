"""
This module defines a base class for probability distribution functions (PDFs)
of a discrete or continuous scalar variate.

We assume the implicit existence of an invertible "link" function that maps the
distributional parameters into an internal representation involving a scalar
link parameter and a complementary vector of independent parameters (if any are
required). Each such internal parameter will have a corresponding variate with
a mean and variance, and these internal variates will have covariances.

This module also defines a base class for conditional PDFs that depend upon
covariates via a regression model with regression parameters. The basic
assumption is that the conditional distribution is controlled by the link
parameter, which in turn is determined by the regression model.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional
import numpy as np
from numpy import ndarray
from numpy.linalg import solve


"""
A Value represents the 'value' of a single parameter or variate, which may in
fact be either single-valued (i.e. scalar) or multi-valued (i.e. an array of
different values).
"""
Value = Union[float, ndarray]
"""
A Values represents the 'value(s)' of one or more parameters or variates
(either single-valued or multi-valued) in some fixed order.
"""
Values = Tuple[Value]
"""
A Scalars represents the scalar value(s) of one or more parameters or
variates in some fixed order.
"""
Scalars = Tuple[float]


###############################################################################
# Useful functions for regression:


def add_intercept(column, *columns) -> ndarray:
    """
    Wraps the given vector(s) into a column matrix suitable for covariates
    in a RegressionPDF. The first column is an additional unit vector
    representing an intercept or bias term in the regresson model.

    Inputs:
        - column (array-like): The values of the first covariate.
        - columns (tuple of array-like): The optional values of subsequent
            covariates.
    Returns:
        - matrix (ndarray): The two-dimensional covariate matrix.
    """
    const = np.ones(len(column))
    return np.stack((const, column) + columns, axis=1)


def no_intercept(column, *columns) -> ndarray:
    """
    Wraps the given vector(s) into a column matrix suitable for covariates
    in a RegressionPDF.

    Inputs:
        - column (array-like): The values of the first covariate.
        - columns (tuple of array-like): The optional values of subsequent
            covariates.
    Returns:
        - matrix (ndarray): The two-dimensional covariate matrix.
    """
    return np.stack((column,) + columns, axis=1)


def is_scalar(*params: Values) -> bool:
    """
    Determines whether or not the given parameters all have valid, scalar
    values.

    Input:
        - params (tuple of float or ndarray): The parameter values.
    Returns:
        - flag (bool): A value of true if scalar-valued, otherwise False.
    """
    for p in params:
        if isinstance(p, ndarray):
            return False  # Multi-valued
        if np.isnan(p):
            return False  # Divergent parameters
    return True


###############################################################################
# Base distribution class:


class BasePDF(ABC):
    """
    A probability distribution of a scalar variate, X.
    """

    def __init__(self, *theta: Values):
        """
        Initialises the distribution(s).

        Each parameter may have either a single value or multiple values.
        If all parameters are single-valued, then only a single distribution
        is specified, and all computations, e.g. the distributional mean or
        variance, etc., will be single-valued.

        However, the use of one or more parameters with multiple values
        indicates a collection of distributions, rather than a single
        distribution. As such, all computations will be multi-valued
        rather than single-valued.

        Input:
            - theta (tuple of float or ndarray): The parameter value(s).
        """
        self.set_parameters(*theta)

    def set_parameters(self, *theta: Values):
        """
        Initialises the distributional parameter value(s).

        Input:
            - theta (tuple of float or ndarray): The parameter value(s).
        """
        params = []
        size = 1
        for i, param in enumerate(theta):
            if isinstance(param, ndarray):
                if len(param.shape) != 1:
                    raise ValueError(f"Expected parameter {i} to be uni-dimensional")
                _len = len(param)
                if _len <= 0:
                    raise ValueError(f"Expected parameter {i} to be non-empty")
                elif _len == 1:
                    # Take scalar value
                    param = param[0]
                elif size == 1:
                    size = _len
                elif size != _len:
                    raise ValueError(f"Expected parameter {i} to have length {size}")
            params.append(param)
        self._params = tuple(params)
        self._size = size

    def __len__(self):
        """
        Determines the number of distributions represented by this instance.

        Returns:
            - length (int): The number of distributions.
        """
        return self._size

    def reset_parameters(self):
        """
        Resets the distributional parameters to their default (scalar) values.
        """
        self.set_parameters(*self.default_parameters())

    @abstractmethod
    def default_parameters(self) -> Scalars:
        """
        Provides default (scalar) values of the distributional parameters.

        Returns:
            - theta (tuple of float): The default parameter values.
        """
        raise NotImplementedError

    def is_scalar(self) -> bool:
        """
        Determines whether or not the distribution has valid, scalar-valued
        parameters.

        Returns:
            - flag (bool): A value of True if scalar-valued, otherwise False.
        """
        if len(self) > 1:
            return False  # Multi-valued
        return is_scalar(*self.parameters())

    def is_default(self) -> bool:
        """
        Determines whether or not the distribution has default, scalar-valued
        parameters.

        Returns:
            - flag (bool): A value of True if default-valued, otherwise False.
        """
        if len(self) > 1:
            return False  # Multi-valued
        for p, d in zip(self.parameters(), self.default_parameters()):
            if p != d:
                return False  # Not default
        return True

    # ----------------------
    # Standard PDF methods:

    def parameters(self) -> Values:
        """
        Provides the values of the distributional parameters.

        Returns:
            - theta (tuple of float or ndarray): The parameter values.
        """
        return self._params

    @abstractmethod
    def mean(self) -> Value:
        """
        Computes the mean(s) of the distribution(s).

        Returns:
            - mu (float or ndarray): The mean value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Value:
        """
        Computes the variance(s) of the distribution(s).

        Returns:
            - sigma_sq (float or ndarray): The variance(s).
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, X: Value) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
        Returns:
            - log_prob (float or ndarray): The log-likelihood(s).
        """
        raise NotImplementedError

    # -------------------------------------------
    # Reparameterisation via link parameter + independent parameters:

    @abstractmethod
    def link_parameters(self) -> Values:
        """
        Computes the value(s) of the link parameter, and optionally the
        value(s) of any independent parameters.
        The link parameter value(s) will always be located at index 0.

        This is a generalisation of the GLM link function, which now maps the
        distributional parameters into internal parameters.

        Returns:
            - params (tuple of float or ndarray): The internal parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def link_inversion(self, eta: Value, *psi: Values) -> Values:
        """
        Computes the value(s) of the distributional parameters from the given
        value(s) of the link parameter and the independent parameters (if any).

        Note that if the independent parameters exist but are not specified,
        then their current values will be used instead.

        This is a generalisation of the inverse GLM link function, which now
        maps the internal parameters into distributional parameters.

        Input:
            - eta (float or ndarray): The link parameter value(s).
            - psi (tuple of float or ndarray): The independent parameter values.
        Returns:
            - theta (tuple of float or ndarray): The distributional parameter
                value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def link_variates(self, X: Value) -> Values:
        """
        Computes the value(s) of the link variate, and optionally the
        value(s) of any independent variates.
        The link variate value(s) will always be located at index 0.

        Input:
            - X (float or ndarray): The value(s) of the distributional variate.

        Returns:
            - Y (tuple of float or ndarray): The internal variate values.
        """
        raise NotImplementedError

    @abstractmethod
    def link_means(self) -> Values:
        """
        Computes the mean(s) of the link variate, and optionlly the mean(s) of
        any independent variates.
        The link mean value(s) will always be located at index 0.

        Returns:
            - mu (tuple of float or ndarray): The internal variate means.
        """
        raise NotImplementedError

    @abstractmethod
    def link_variances(self) -> ndarray:
        """
        Computes the variance(s) of the link variate, and optionally the
        variances and covariances of all independent variates, and the
        covariances between the link variate and the independent variates.
        The link variance value(s) will always be located at index [0, 0].

        Returns:
            - Sigma (ndarray): The variances and covariances of all internal
                variates.
        """
        raise NotImplementedError

    def fit(
        self,
        X: Value,
        W: Optional[Value] = None,
        max_iters: int = 100,
        min_tol: float = 1e-6,
        step_size: float = 1.0,
    ) -> Tuple[float, int, float]:
        """
        Re-estimates the PDF parameters from the given observation(s).

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - W (float or ndarray): The optional weight(s) of the covariates.
            - max_iters (float): The maximum number of iterations; defaults to 100.
            - min_tol (float): The minimum score tolerance to indicate convergence;
                defaults to 1e-6.
            - step_size (float): Scales the parameter update. Reduce this value
                if necessary to avoid divergence. Defaults to 1.0.
        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - tol (float): The final score tolerance.
        """
        # Enforce a single distribution, i.e. scalar parameter values
        if not self.is_scalar():
            self.reset_parameters()
        # Allow for single or multiple observations
        if isinstance(X, ndarray):
            # Obtain a weight for each observation
            if W is None:
                W = np.ones(len(X))
            elif not isinstance(W, ndarray) or len(W) != len(X):
                raise ValueError("Incompatible weights!")
            tot_W = np.sum(W)
            # Specify weighted mean function
            mean_fn = lambda t: (W @ np.column_stack(t)) / tot_W
        else:
            # No weights or averaging required
            mean_fn = np.array
            tot_W = W = 1.0
        # Obtain current score
        score = np.sum(W * self.log_prob(X)) / tot_W
        tol = 0.0
        # Fit data
        num_iters = 0
        while num_iters < max_iters:
            num_iters += 1
            # Update internal parameters
            Y = mean_fn(self.link_variates(X))
            mu = np.array(self.link_means())
            vars = self.link_variances()
            delta = solve(vars, Y - mu)
            params = np.array(self.link_parameters()) + step_size * delta
            # Update distributional parameters
            self.set_parameters(*self.link_inversion(*params))
            # Obtain new score
            new_score = np.sum(W * self.log_prob(X)) / tot_W
            tol = new_score - score
            score = new_score
            if np.abs(tol) < min_tol:
                break
        return score, num_iters, tol


###############################################################################
# Regression class:


class RegressionPDF(ABC):
    """
    A conditional probability distribution of a scalar variate, X, that depends
    upon a regression model involving covariates, Z.
    """

    def __init__(self, pdf: BasePDF, phi: Optional[ndarray] = None):
        """
        Initialises the conditional distribution using an underlying
        unconditional distribution.

        The independent parameters are obtained from the underlying distribution.
        If necessary, the distributional parameters may be reset to take their
        default values, in order to ensure that the independent parameters
        have valid, scalar values.

        The regression model will override the value(s) of the link parameter.
        Note that if the regression model parameters are not supplied here,
        then they must be obtained by fitting the conditional distribution to
        observed data.

        Input:
            - pdf (BasePDF): The underlying probability distribution.
            - phi (ndarray): The optional regression model parameters.
        """
        self._pdf = pdf
        self.reset_parameters()
        self._reg_params = phi

    def distribution(self) -> BasePDF:
        """
        Obtains the underlying (unconditional) distribution.

        Returns:
            - pdf (BasePDF): The underlying distribution.
        """
        return self._pdf

    def independent_parameters(self) -> ndarray:
        """
        Provides the values of the independent distributional parameters.

        Returns:
            - psi (ndarray): The (possibly empty) parameter values.
        """
        return self._indep_params

    def regression_parameters(self) -> ndarray:
        """
        Provides the values of the regression model parameters.

        Returns:
            - phi (ndarray): An array of parameter values.
        """
        phi = self._reg_params
        if phi is None:
            raise ValueError("Regression parameters have not been specified!")
        return phi

    def _init_regression_parameters(self, num_params: int):
        """
        Initialises default values of the regression parameters.

        Input:
            - num_params (int): The number of regression parameters.
        """
        self._reg_params = (1e-2 / num_params) * np.ones(num_params)

    def is_scalar(self) -> bool:
        """
        Determines whether or not the conditional distribution has valid,
        scalar-valued parameters.

        Returns:
            - flag (bool): A value of True if scalar-valued, otherwise False.
        """
        if not is_scalar(*self._indep_params):
            return False
        if self._reg_params is not None:
            return is_scalar(self._reg_params)
        return True

    def reset_parameters(self):
        """
        Resets the independent parameter values, which may involve resetting
        the underlying distribution, if necessary.
        Also removes the regression parameters, which will need to be refitted.
        """
        if not is_scalar(*self._pdf.link_parameters()[1:]):
            self._pdf.reset_parameters()
        self._indep_params = np.array(self._pdf.link_parameters()[1:], dtype=float)
        self._reg_params = None

    def log_prob(self, X: Value, Z: ndarray) -> Value:
        """
        Computes the conditional log-likelihood(s) of the given data.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - Z (ndarray): The value(s) of the covariates.
        Returns:
            - log_prob (float or ndarray): The log-likelihood(s).
        """
        eta = self._regression_value(Z, self.regression_parameters())
        self._pdf.set_parameters(*self._pdf.link_inversion(eta))
        return self._pdf.log_prob(X)

    def mean(self, Z: ndarray) -> Value:
        """
        Computes the conditional mean(s) for the given covariates.

        Inputs:
            - Z (ndarray): The value(s) of the covariates.
        Returns:
            - mu (float or ndarray): The predicted mean(s).
        """
        eta = self._regression_value(Z, self.regression_parameters())
        self._pdf.set_parameters(*self._pdf.link_inversion(eta))
        return self._pdf.mean()

    def _regression_value(self, Z: ndarray, phi: ndarray) -> Value:
        """
        Evaluates the regression function at the covariate value(s).

        Input:
            - Z (ndarray): The value(s) of the covariates.
            - phi (ndarray): The regression model parameters.
        Returns:
            - f (float or ndarray): The function value(s).
        """
        # Assume a linear model for convenience
        return Z @ phi

    def _regression_gradient(self, Z: ndarray, phi: ndarray) -> Value:
        """
        Evaluates the gradient of the regression function at the covariate value(s).

        Input:
            - Z (ndarray): The value(s) of the covariates.
            - phi (ndarray): The regression model parameters.
        Returns:
            - g (ndarray): The gradient value(s).
        """
        # Assume a linear model for convenience
        return Z

    def _compute_score(
        self, X: Value, Z: ndarray, W: Value, phi: ndarray, *psi: Values
    ) -> float:
        """
        Updates the distributional parameters and computes the log-likelihood score.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - Z (ndarray): The value(s) of the explanatory covariates.
            - W (float or ndarray): The weight(s) of the covariates.
            - phi (ndarray): The regression parameter values.
            - psi (tuple of float or ndarray): The optional independent parameter values.
        Returns:
            - score (float): The score.
        """
        eta = self._regression_value(Z, phi)
        self._pdf.set_parameters(*self._pdf.link_inversion(eta, *psi))
        score = np.sum(W * self._pdf.log_prob(X)) / np.sum(W)
        return score

    def _compute_deltas(
        self, X: ndarray, Z: ndarray, W: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        Computes the updates (deltas) for the regression model parameters
        and the independent parameters (if any).

        Inputs:
            - X (ndarray): The values of the response variate.
            - Z (ndarray): The values of the explanatory covariates.
            - W (ndarray): The weights of the covariates.
        Returns:
            - delta_phi (ndarray): The regression parameters update vector.
            - delta_psi (ndarray): The independent parameters update vector.
        """
        # The task is to solve the matrix equation:
        #   [v_phi cov^T] * [d_phi] = [r_phi]
        #   [cov   v_psi]   [d_psi]   [r_psi]
        # using block matrix inversion. An answer (for v_phi invertible) is:
        #   d_psi = S^-1 * (r_psi - cov * v_phi^-1 * r_phi),
        #   d_phi = v_phi^-1 * (r_phi - cov^T * d_psi),
        # where S is the Schur complement:
        #   S = v_psi - cov * v_phi^-1 * cov^T.

        # Obtain distributional info
        Y = self._pdf.link_variates(X)
        mu = self._pdf.link_means()
        vars = self._pdf.link_variances()
        # Extract link parameter info
        Y_eta = Y[0]
        mu_eta = mu[0]  # E[Y_eta]
        v_eta = vars[0, 0]  # Var[Y_eta]
        # Compute regression Y_phi, E[Y_phi] and Var[Y_phi]
        N = len(X)
        v_phi = sum(W[k] * v_eta[k] * np.outer(Z[k, :], Z[k, :]) for k in range(N))
        r_phi = (W * (Y_eta - mu_eta)) @ Z
        # Inversion step 1: d_phi' = v_phi^-1 * r_phi
        d_phi = solve(v_phi, r_phi)
        if len(Y) == 1:
            # No independent parameters - just compute delta_phi
            return d_phi, np.array([])
        # Compute independent Y_psi, E[Y_psi] and Var[Y_psi]
        Y_psi = np.column_stack(Y[1:])
        mu_psi = np.column_stack(mu[1:])  # E[Y_psi]
        r_psi = W @ (Y_psi - mu_psi)
        v_psi = vars[1:, 1:]  # Var[Y_psi]
        v_psi = sum(W[k] * v_psi[:, :, k] for k in range(N))
        # Extract covariances between link and independent parameters
        cov = vars[0, 1:]  # Cov[Y_eta, Y_psi]
        # Convert to regression covariance, Cov[Y_psi, Y_phi]
        cov = sum(W[k] * np.outer(cov[:, k], Z[k, :]) for k in range(N))
        # Inversion step 2: S = v_psi - cov * v_phi^-1 * cov^T
        S = v_psi - cov @ solve(v_phi, cov.T)
        # Inversion step 3: d_psi = S^-1 * (r_psi - cov * d_phi')
        d_psi = solve(S, r_psi - cov @ d_phi)
        # Inversion step 4: d_phi = d_phi' - v_phi^-1 * cov^T * d_psi
        d_phi -= solve(v_phi, cov.T @ d_psi)
        return d_phi, d_psi

    def fit(
        self,
        X: Value,
        Z: ndarray,
        W: Optional[Value] = None,
        max_iters: int = 100,
        min_tol: float = 1e-6,
        step_size: float = 1.0,
    ) -> Tuple[float, int, float]:
        """
        Re-estimates the regression model parameters and the independent
        distributional parameters from the given observation(s).

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - Z (ndarray): The value(s) of the explanatory covariates.
            - W (float or ndarray): The optional weight(s) of the covariates.
            - max_iters (float): The maximum number of iterations.
            - min_tol (float): The minimum score tolerance to indicate convergence.
            - step_size (float): Scales the parameter update. Reduce this value
                if necessary to avoid divergence.
        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - tol (float): The final score tolerance.
        """
        # Enforce a single distribution, i.e. scalar parameter values
        if not self.is_scalar():
            self.reset_parameters()
        # Allow for single or multiple observations
        if not isinstance(Z, ndarray):
            raise ValueError("Incompatible covariates!")
        if isinstance(X, ndarray):
            # Multiple observations
            if len(X.shape) != 1:
                raise ValueError("Incompatible variates!")
            if len(Z.shape) != 2 or Z.shape[0] != len(X):
                raise ValueError("Incompatible covariates!")
        else:
            # Single observation - convert to multiple form
            X = np.array([X])
            if len(Z.shape) != 1:
                raise ValueError("Incompatible covariates!")
            Z = Z.reshape((1, -1))
        # Obtain a weight for each observation
        if W is None or not isinstance(W, ndarray):
            W = np.ones(len(X))
        elif len(W.shape) != 1 or W.shape[0] != len(X):
            raise ValueError("Incompatible weights!")
        tot_W = np.sum(W)
        # Check regression parameters
        num_params = Z.shape[1]
        if self._reg_params is None:
            self._init_regression_parameters(num_params)
        elif len(self._reg_params) != num_params:
            raise ValueError("Incompatible regression parameters!")
        phi = self.regression_parameters()
        # Obtain current score
        score = self._compute_score(X, Z, W, phi)
        tol = 0.0
        # Fit data
        num_iters = 0
        while num_iters < max_iters:
            num_iters += 1
            Z0 = self._regression_gradient(Z, phi)
            d_phi, d_psi = self._compute_deltas(X, Z0, W)
            # Update regression parameter estimates
            phi += step_size * d_phi
            # Update independent parameters (if any)
            psi = self.independent_parameters()
            if len(psi) > 0:
                psi += step_size * d_psi
            # Update distributional parameters and score
            new_score = self._compute_score(X, Z, W, phi, *psi)
            tol = new_score - score
            score = new_score
            if np.abs(tol) < min_tol:
                break
        return score, num_iters, tol


class ScalarPDF(BasePDF):
    """
    Enables the distribution to be wrapped into a conditional PDF.
    """

    def regressor(self, phi: Optional[ndarray] = None) -> RegressionPDF:
        """
        Wraps the distribution into a conditional PDF, which evaluates the
        link parameter via a regression function of covariates.
        Note that the regressor may modify the distributional parameters.

        Input:
            - phi (ndarray): The optional regression model parameters.
        Returns:
            - reg (RegressionPDF): The conditional PDF.
        """
        return RegressionPDF(self, phi)
