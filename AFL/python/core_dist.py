"""
This module defines a base class for probability distribution functions (PDFs)
of a discrete or continuous scalar variate.

We assume for convenience that every PDF p(X|theta) is a member of "the" exponential family,
and thus has natural parameters (eta) and natural variates (Y) that might not be the same
as the default parameters (theta) and the given variate (X).

We also assume the implicit existence of an invertible link function (g) that maps the mean
mu of the distribution into a so-called link parameter (also eta). Tne derivative of the
log-likehood function with respect to this link parameter defines the difference between
the so-called link variate (Y_eta) and its mean (mu_eta). Note that the link parameter is
not necessarily related to the natural distributional parameters, despite traditionally
having the same label (i.e. both are called eta).

The module also defines a base class for PDFs that depend upon covariates (Z)
via a regression model with parameters (phi). The basic assumption is that it is
the link parameter (eta) that is explained by the regression model.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np
from numpy import ndarray
from numpy.linalg import solve


# A Value represents one parameter or variate, which may be single-valued
# or multi-valued.
Value = Union[float, ndarray]
# A Collection represents one or more parameters or variates.
Collection = Tuple[Value]


class ScalarPDF(ABC):
    """
    A probability distribution of a scalar variate, X.
    """

    def __init__(self, *theta):
        """
        Initialises the distribution.

        Each parameter may have either a single value or multiple values.
        If all parameters are single-valued, then only a single distribution
        is specified, and all computations, e.g. the distributional mean or
        variance, etc., will be single-valued.

        However, the use of one or more parameters with multiple values
        indicates a collection of distributions, rather than a single
        distribution. As such, all computations will be multi-valued
        rather than single-valued.

        Input:
            - theta (array of float or ndarray): The parameter values.
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
        Determines whether the instance represents a single distribution
        or multiple distributions.

        Returns:
            - length (int): The number of distributions represented.
        """
        return self._size

    def parameters(self) -> Collection:
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
    def natural_parameters(self) -> Collection:
        """
        Computes the values of the natural parameters.

        Returns:
            - eta (tuple of float or ndarray): The natural parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_variates(self, X: Value) -> Collection:
        """
        Computes the values of the natural variates from the given variate.

        Input:
            - X (float or ndarray): The value(s) of the distributional variate.
        Returns:
            - Y (tuple of float or ndarray): The natural variate values.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_means(self) -> Collection:
        """
        Computes the means of the natural variates.

        Returns:
            - mu (tuple of float or ndarray): The natural variate means.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_variances(self) -> ndarray:
        """
        Computes the variances and covariances of the natural variates.

        Returns:
            - Sigma (ndarray): The variance matrix (or tensor) of the
                natural variates.
        """
        raise NotImplementedError

    @abstractmethod
    def link_parameter(self) -> Value:
        """
        Computes the value(s) of the link parameter from the mean(s) via the
        link function.

        Returns:
            - eta (float or ndarray): The link parameter value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def link_variate(self, X: Value) -> Value:
        """
        Computes the value(s) of the link variate from the given variate.

        Input:
            - X (float or ndarray): The value(s) of the distributional variate.
        Returns:
            - Y (float or ndarray): The link variate value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def link_mean(self) -> Value:
        """
        Computes the mean(s) of the link variate.

        Returns:
            - mu (float or ndarray): The link variate mean(s).
        """
        raise NotImplementedError

    @abstractmethod
    def link_variance(self) -> Value:
        """
        Computes the variance(s) of the link variate.

        Returns:
            - sigma_sq (float or ndarray): The link variate variance(s).
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


class RegressionPDF(ABC):
    """
    A probability distribution of a scalar variate, X, for which some or all of
    the distributional parameters depend upon a regression model involving
    covariates, Z. Any distributional parameters not depending upon the
    regression model are treated as independent parameters.
    """

    def __init__(self, phi: ndarray, *theta):
        """
        Initialises the distribution.

        Input:
            - phi (ndarray): The regression model parameters.
            - theta (array of float): The independent distributional parameters.
        """
        self._reg_params = phi
        self._params = np.array(theta)

    def regression_parameters(self) -> ndarray:
        """
        Provides the values of the regression model parameters.

        Returns:
            - phi (ndarray): An array of parameter values.
        """
        return self._reg_params

    def independent_parameters(self) -> ndarray:
        """
        Provides the values of the independent distributional parameters.

        Returns:
            - theta (array): A (possibly empty) array of parameter values.
        """
        return self._params

    def link_parameter(self, Z: ndarray) -> Value:
        """
        Computes the value(s) of the link parameter from the regression model.

        Input:
            - Z (ndarray): The value(s) of the covariates.
        Returns:
            - eta (float or ndarray): The link parameter value(s).
        """
        # Assume a linear (or affine) model for convenience
        return Z @ self.regression_parameters()

    @abstractmethod
    def _distribution(self, eta: Value) -> ScalarPDF:
        """
        Creates a distribution (or a collection of distributions) from the
        independent parameters and the value(s) of the link parameter.

        Input:
            - eta (float or ndarray): The link parameter value(s).
        Returns:
            - pdf (ScalarPDF): The distribution(s).
        """
        raise NotImplementedError

    def distribution(self, Z: ndarray) -> ScalarPDF:
        """
        Creates a distribution (or a collection of distributions) from the
        value(s) of the covariates.

        Input:
            - Z (ndarray): The value(s) of the covariates.
        Returns:
            - pdf (ScalarPDF): The distribution(s).
        """
        eta = self.link_parameter(Z)
        return self._distribution(eta)

    def _regression_delta(self, X: Value, Z: ndarray, pdf: ScalarPDF) -> ndarray:
        """
        Computes the update (delta) for the regression model parameters.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - Z (ndarray): The value(s) of the explanatory covariates.
            - pdf (ScalarPDF): The regression distribution(s).
        Returns:
            - delta (ndarray): The update vector.
        """
        Y = pdf.link_variate(X)
        mu = pdf.link_mean()
        sigma_sq = pdf.link_variance()
        if len(Z.shape) == 2:
            # Multiple data
            N = Z.shape[0]
            lhs = sum(
                [sigma_sq[k] * np.outer(Z[k,:], Z[k,:]) for k in range(N)]
            ) / N
            rhs = (X - mu) @ Z / N
        else:
            # Single datum
            lhs = sigma_sq * np.outer(Z, Z)
            rhs = (X - mu) * Z
        return solve(lhs, rhs)

    @abstractmethod
    def _independent_delta(self, X: Value, pdf: ScalarPDF) -> ndarray:
        """
        Computes the update (delta) for the independent distributional parameters.

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - pdf (ScalarPDF): The regression distribution(s).
        Returns:
            - delta (ndarray): The (possibly empty) update vector.
        """
        raise NotImplementedError

    def fit(self, X: Value, Z: ndarray, max_iters: int = 100, min_tol: float = 1e-6) -> Tuple[float, int, float]:
        """
        Re-estimates the regression model parameters and the independent
        distributional parameters from the given observation(s).

        Inputs:
            - X (float or ndarray): The value(s) of the response variate.
            - Z (ndarray): The value(s) of the explanatory covariates.
            - max_iters (float): The maximum number of iterations; defaults to 100.
            - min_tol (float): The minimum score tolerance to indicate convergence;
                defaults to 1e-6.
        Returns:
            - score (float): The mean log-likelihood of the data.
            - num_iters (int): The number of iterations performed.
            - tol (float): The final score tolerance.
        """
        # Obtain current score
        pdf = self.distribution(Z)
        score0 = np.mean(pdf.log_prob(X))
        tol = 0.0
        # Fit data
        N = len(X)
        num_iters = 0
        while num_iters < max_iters:
            num_iters += 1
            # Update regression parameter estimates
            delta_phi = self._regression_delta(X, Z, pdf)
            phi = self.regression_parameters()
            phi += delta_phi
            # Update independent parameter estimates
            theta = self.independent_parameters()
            if len(theta) > 0:
                delta_theta = self._independent_delta(X, pdf)
                theta += delta_theta
            # Obtain new score
            pdf = self.distribution(Z)
            score1 = np.mean(pdf.log_prob(X))
            tol = score1 - score0
            score0 = score1
            if np.abs(tol) < min_tol:
                break
        return score0, num_iters, tol
