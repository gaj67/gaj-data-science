"""
This module defines base classes for a probability distribution function (PDF)
of a discrete or continuous, scalar variate.

Typically, a distribution will also be parameterised, but this is not required.
For example, the logistic distribution is not parameterised in its standard form.
"""

from abc import ABC, abstractmethod

from .data_types import (
    Value,
    Values,
    Vector,
    VectorLike,
    MatrixLike,
)

from .regressor import Regression


###############################################################################
# Base distribution classes:


class Distribution(ABC):
    """
    Encapsulates one or more probability distributions from a family of
    distributions of a scalar variate, X.

    Note that if only a single distribution is represented, then
    the mean() and variance() will be scalar valued. However, for
    multiple distributions, these values will be vectors.
    """

    @abstractmethod
    def mean(self) -> Value:
        """
        Obtains the mean(s) of the distribution(s).

        Returns:
            - mu (float-like or vector): The mean value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Value:
        """
        Obtains the variance(s) of the distribution(s).

        Returns:
            - sigma_sq (float-like or vector): The variance(s).
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, variate: VectorLike) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - variate (vector-like): The value(s) of the response variate.

        Returns:
            - log_prob (float or vector): The log-likelihood(s).
        """
        raise NotImplementedError


###############################################################################
# Base conditional distribution class:


class ConditionalDistribution(ABC):
    """
    Encapsulates a conditional probability distribution of a scalar variate, X,
    dependent on one or more covariates, Z.

    Note that if only a single observation of the covariate(s) is given, then
    the mean() and variance() will be scalar valued. However, for multiple
    observations, these values will be vectors.
    """

    @abstractmethod
    def mean(self, covariates: MatrixLike) -> Value:
        """
        Obtains the conditional mean(s) of the distribution(s).

        Input:
            - covariates (matrix-like): The covariate value(s).

        Returns:
            - mu (float-like or vector): The mean value(s).
        """
        raise NotImplementedError

    @abstractmethod
    def variance(self, covariates: MatrixLike) -> Value:
        """
        Obtains the conditional variance(s) of the distribution(s).

        Input:
            - covariates (matrix-like): The covariate value(s).

        Returns:
            - sigma_sq (float-like or vector): The variance(s).
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - variate (vector-like): The value(s) of the response variate.
            - covariates (matrix-like): The covariate value(s).

        Returns:
            - log_prob (float or vector): The log-likelihood(s).
        """
        raise NotImplementedError


###############################################################################
# Base regression distribution class:


class RegressionDistribution(Regression, ConditionalDistribution):
    """
    Implements a conditional distribution using regression to compute
    the parameters of an underlying multi-distribution.
    """

    def __init__(self, pdf: Distribution, reg_params: Vector, *indep_params: Values):
        """
        Initialises the conditional distribution.

        Use the UNSPECIFIED_REGRESSION constant to indicate that the regression
        parameters are unknown, and need to be set or estimated.

        Input:
            - pdf (distribution): The underlying probability distribution.
            - reg_params (vector): The value(s) of the regression parameter(s).
            - indep_params (tuple of float, optional): The value(s) of the
                independent parameter(s), if any.
        """
        super().__init__(reg_params, *indep_params)
        self._pdf = pdf

    def distribution(self) -> Distribution:
        """
        Obtains the underlying distribution.

        Returns:
            - pdf (distribution): The distribution instance.
        """
        return self._pdf

    def mean(self, covariates: MatrixLike) -> Value:
        self.invert_regression(covariates)
        return self.distribution().mean()

    def variance(self, covariates: MatrixLike) -> Value:
        self.invert_regression(covariates)
        return self.distribution().variance()

    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        self.invert_regression(covariates)
        return self.distribution().log_prob(variate)

    def invert_regression(self, covariates: MatrixLike) -> Values:
        """
        Computes the regression function and then inverts the
        link parameter (and any independent parameters) into parameters
        of the underlying distribution.

        Input:
            - covariates (matrix-like): The covariate value(s).
        """
        link_param = self.apply_regression(covariates)
        params = self.invert_link(link_param, *self.independent_parameters())
        self.distribution().set_parameters(*params)
