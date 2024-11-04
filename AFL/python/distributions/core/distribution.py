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
    Values2d,
    Vector,
    VectorLike,
    MatrixLike,
    is_scalar,
    is_vector,
    is_divergent,
    to_vector,
    as_value,
    mult_rmat_vec,
    mult_rmat_rmat,
)

from .parameterised import Parameterised, Parameters, RegressionParameters  # ??
from .fittable import Fittable, GradientOptimisable, TransformOptimisable
from .regressable import Linkable, Regressable, RegressionOptimisable


###############################################################################
# Base distribution classes:


class Distribution(ABC):
    """
    An interface for one or more probability distributions from a family of
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


class StandardDistribution(Distribution, GradientOptimisable, Fittable):
    """
    Implements a parameterised distribution with an optimisable
    log-likelihood score.
    """

    # Distribution interface

    def log_prob(self, variate: VectorLike) -> Value:
        return as_value(self.compute_scores(to_vector(variate)))


###############################################################################
# Transformed distribution class:


class TransformDistribution(StandardDistribution, TransformOptimisable):
    """
    Encapsulates a distribution using an alternative parameterisation
    to the standard form.

    There exists an invertible transformation between the standard
    parameterisation of the underlying distribution and the alternative
    parameterisation.

    Note: The transformed distribution does not itself hold any parameters.
    """

    def __init__(self, dist: StandardDistribution, *params: Values):
        """
        Encapsulates an instance of the specified distribution, and initialises
        the values of the alternative parameters.

        Input:
            - dist (distribution): An instance of a parameterised distribution.
            - params (tuple of float or vector): The value(s) of the alternative
                parameter(s).
        """
        self._dist = dist
        if len(params) == 0:
            params = self.default_parameters()
        self.set_parameters(*params)

    def underlying(self) -> StandardDistribution:
        """
        Obtains the underlying distribution.

        Returns:
            - dist (distribution): The distribution instance.
        """
        return self._dist

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        return self.underlying().mean()

    def variance(self) -> Value:
        return self.underlying().variance()

    def log_prob(self, variate: VectorLike) -> Value:
        return self.underlying().log_prob(variate)


###############################################################################
# Base regression distribution classes:


class ConditionalDistribution(ABC):
    """
    An interface for a conditional probability distribution of
    a scalar variate, X, dependent on one or more covariates, Z.

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


class RegressionDistribution(
    RegressionParameters, ConditionalDistribution, RegressionOptimisable, Regressable
):
    """
    Implements a conditional distribution using linear regression to compute
    the parameters of the underlying distribution.
    """

    def __init__(self, link: StandardDistribution, *params: Values):
        """
        Initialises the conditional distribution using a regression model
        with an underlying link distribution.

        The model parameters are assumed to consist of a vector of regression
        parameters, optionally followed by any independent parameters.

        The link distribution is assumed to be parameterised by a link parameter,
        optionally followed by the independent parameters.

        Input:
            - dist (distribution): The underlying link distribution.
            - params (tuple of scalar or vector): The value(s) of the
                regression parameter(s), and independent parameter(s), if any.
        """
        RegressionParameters.__init__(self, link, *params)

    def underlying(self) -> Distribution:
        """
        Obtains the underlying link distribution.

        Returns:
            - dist (distribution): The distribution instance.
        """
        return RegressionParameters.underlying(self)

    # ---------------------------------
    # ConditionalDistribution interface

    def mean(self, covariates: MatrixLike) -> Value:
        self._invert_regression(covariates)
        return self.underlying().mean()

    def variance(self, covariates: MatrixLike) -> Value:
        self._invert_regression(covariates)
        return self.underlying().variance()

    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        self._invert_regression(covariates)
        return self.underlying().log_prob(variate)
