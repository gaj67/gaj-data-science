"""
This module defines base classes for a probability distribution function (PDF)
of a discrete or continuous, scalar variate.

Typically, a distribution will also be parameterised, but this is not required.
For example, the logistic distribution is not parameterised in its standard form.
"""

from abc import ABC, abstractmethod
from typing import Callable, Type

from .data_types import (
    Value,
    Values,
    VectorLike,
    MatrixLike,
    to_vector,
    as_value,
)

from .parameterised import (
    Parameterised,
    Parameters,
    OneVectorParameters,
)
from .fittable import Fittable, GradientOptimisable, TransformOptimisable
from .regressable import Regressable, RegressionOptimisable
from .optimiser import Controls, set_controls


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


class BaseDistribution(Distribution, GradientOptimisable, Fittable):
    """
    Interface for a parameterised distribution with an optimisable
    log-likelihood score.

    Note: The implementation of this interface must hold the parameters.
    """

    # Distribution interface

    def log_prob(self, variate: VectorLike) -> Value:
        # Assume the objective function is the log-likelihhood
        return as_value(self.compute_scores(to_vector(variate)))


###############################################################################
# Transformed distribution class:


class TransformDistribution(TransformOptimisable, BaseDistribution):
    """
    Encapsulates an underlying distribution with an alternative parameterisation.

    There exists an invertible transformation between the standard
    parameterisation of the underlying distribution and the alternative
    parameterisation.

    Note: The transformed distribution does not itself hold any parameters.
    Parameters are held by the implementation of the underlying distribution.
    """

    def __init__(self, dist: BaseDistribution):
        """
        Encapsulates an instance of the specified distribution.

        Input:
            - dist (distribution): An instance of a parameterised distribution.
        """
        self._dist = dist

    def underlying(self) -> BaseDistribution:
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
    ConditionalDistribution, RegressionOptimisable, Regressable
):
    """
    A partial implementation for a conditional distribution using
    linear regression to optimise the parameters of the underlying
    link distribution.
    """

    def __init__(self, link: BaseDistribution):
        """
        Initialises the conditional distribution using a regression model
        with an underlying link distribution.

        The model parameters are assumed to consist of a vector of regression
        parameters, followed by any additional parameters required by the link
        distribution.

        The link distribution is assumed to be parameterised by a link parameter,
        followed by the remaining independent parameters, if any.

        Input:
            - link (distribution): The underlying link distribution.
        """
        self._regression = OneVectorParameters()
        self._link = link

    # ---------------------------------
    # RegressionParameterised interface

    def regression(self) -> Parameterised:
        return self._regression

    def link(self) -> BaseDistribution:
        """
        Obtains the underlying link distribution.

        Returns:
            - dist (distribution): The distribution instance.
        """
        return self._link

    # ---------------------------------
    # ConditionalDistribution interface

    def mean(self, covariates: MatrixLike) -> Value:
        self._invert_regression(covariates)
        return self.link().mean()

    def variance(self, covariates: MatrixLike) -> Value:
        self._invert_regression(covariates)
        return self.link().variance()

    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        self._invert_regression(covariates)
        return self.link().log_prob(variate)


###############################################################################
# Fundamental distribution implementation:


class StandardDistribution(Parameters, BaseDistribution):
    """
    A partial implementation of a distribution with optimisable
    parameters.
    """

    regressor_klass = RegressionDistribution

    def __init__(self, *params: Values):
        """
        Input:
            - params (tuple of scalar or vector): The value(s) of the
                distributional parameter(s).
        """
        Parameters.__init__(self, *params)

    def link(self) -> BaseDistribution:
        """
        Obtains the link distribution used for regression.

        Returns:
            - link (distribution): The link distribution.
        """
        return self

    def regressor(self) -> RegressionDistribution:
        """
        Obtains a regression distribution for estimating
        parameters from variate and covariate data.

        Returns:
            - regressor (distribution): The regression distribution.
        """
        return self.regressor_klass(self.link())


###############################################################################
# Decorator for specifying the link model:


def set_link(
    link_klass: Type[TransformDistribution],
) -> Callable[[Type[StandardDistribution]], Type[StandardDistribution]]:
    """
    Specifies the link model to underly the regression model.

    Input:
        - link_klass (class): The link model class.

    Returns:
        - decorator (method): A decorator of a distribution class.
    """

    def decorator(klass: Type[StandardDistribution]) -> Type[StandardDistribution]:
        _link_fn = klass.link

        def link(self) -> BaseDistribution:
            return link_klass(_link_fn(self))

        klass.link = link
        klass.link.__doc__ = _link_fn.__doc__
        return klass

    return decorator


###############################################################################
# Decorators for controlling the regression model:


def set_regressor(
    regressor_klass: Type[RegressionDistribution],
) -> Callable[[Type[StandardDistribution]], Type[StandardDistribution]]:
    """
    Sets the class of the regression moodel.

    Input:
        - regressor_klass (class): The regressor class.

    Returns:
        - decorator (method): A distribution class decorator.
    """

    def decorator(klass: Type[StandardDistribution]) -> Type[StandardDistribution]:
        klass.regressor_klass = regressor_klass
        return klass

    return decorator


def set_regressor_controls(
    **controls: Controls,
) -> Callable[[Type[StandardDistribution]], Type[StandardDistribution]]:
    """
    Statically modifies the default values of the controls for the
    regression fitting algorithm.

    Input:
        - controls (dict): The overriden controls and their new default values.
            See Controllable.default_controls().

    Returns:
        - decorator (method): A distribution class decorator.
    """

    def decorator(klass: Type[StandardDistribution]) -> Type[StandardDistribution]:
        klass.regressor_klass = set_controls(**controls)(klass.regressor_klass)
        return klass

    return decorator
