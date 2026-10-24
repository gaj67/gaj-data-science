"""
This module defines base classes for a probability distribution function (PDF)
of a discrete or continuous, scalar variate.

Typically, a distribution will also be parameterised, but this is not required.
For example, the logistic distribution is not parameterised in its standard form.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Type

import numpy as np

from .data_types import (
    Data,
    Value,
    Values,
    Vector,
    VectorLike,
    MatrixLike,
    to_vector,
    as_value,
)

from .parameterised import Parameters
from .controllable import Controls, set_controls
from .fittable import Fittable
from .optimisable import GradientOptimisable, TargetOptimisable, TransformOptimisable
from .regressable import Regressable, RegressionOptimisable


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


# pylint: disable=too-many-ancestors


@set_controls(
    # (bool) Turn on to use least-squares optimisation; AND ...
    least_squares=False,
    # (int) For continuous or many-valued distributions, set to the number of histogram bins
    num_bins=0,
    # (function) Fit least squares in the probability space, by default
    score_fn=np.exp,
)
class BaseDistribution(Distribution, GradientOptimisable, TargetOptimisable, Fittable):
    """
    Interface for a parameterised distribution with a differentiable,
    optimisable log-likelihood score.
    """

    # ----------------------
    # Distribution interface

    def log_prob(self, variate: VectorLike) -> Value:
        # Assume the objective function is the log-likelihhood
        return as_value(self.compute_scores(to_vector(variate)))

    # --------------------------------
    # RegressionDistribution interface

    def link_model(self) -> BaseDistribution:
        """
        Obtains the link distribution used for regression.

        Returns:
            - link_model (distribution): The link distribution.
        """
        return self

    def num_links(self) -> int:
        """
        Obtains the number of leading parameters that will be
        used as link parameters for regression. Any rermaining
        parameters will be used as independent parameters.

        Returns:
            - num (int): The number of link parameters.
        """
        return 0

    # -----------------------------------------------------
    # TargetOptimisable interface for optional least-squares fitting

    def target_scores(self, data: Data, controls: Controls) -> Vector:
        # Estimate empirical distribution
        num_bins = controls.get("num_bins", 0)
        if num_bins > 0:
            # Use unweighted histogram
            _, x_data = np.histogram(data.variate, bins=num_bins)
            x_idx = np.digitize(data.variate, x_data, right=True) - 1
            x_idx[x_idx < 0] = 0
        else:
            # Use unique values
            x_data = np.unique(data.variate)
            x_idx = np.zeros(len(data.variate), dtype=int)
            for i, value in enumerate(x_data):
                x_idx[data.variate == value] = i
            num_bins = len(x_data)
        # Compute weighted contributions
        y_data = np.zeros(num_bins)
        for i in range(num_bins):
            y_data[i] = np.sum(data.weights[x_idx == i])
        # Convert to scores matching compute_scores()
        y_data = np.log(1e-30 + y_data / np.sum(y_data))
        scores = np.zeros(len(data.variate))
        for i, score in enumerate(y_data):
            scores[x_idx == i] = score
        return scores


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

    # ------------------------------
    # TransformOptimisable interface

    def __init__(self, dist: BaseDistribution):
        """
        Encapsulates an instance of the specified distribution.

        Input:
            - dist (distribution): An instance of a parameterised distribution.
        """
        TransformOptimisable.__init__(self, dist)

    def underlying(self) -> BaseDistribution:
        """
        Obtains the underlying optimisable distribution.

        Returns:
            - underlying (distribution): The underlying instance.
        """
        return super().underlying()

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

    # -------------------------------
    # RegressionOptimisable interface

    def __init__(self, link_model: BaseDistribution):
        """
        Initialises the conditional distribution using a regression model
        with an underlying link distribution.

        The model parameters are assumed to consist of a vector of regression
        parameters, followed by any additional parameters required by the link
        distribution.

        The link distribution is assumed to be parameterised by a link parameter,
        followed by the remaining independent parameters, if any.

        Input:
            - link_model (distribution): The underlying link distribution.
        """
        RegressionOptimisable.__init__(self, link_model.num_links(), link_model)

    def link_model(self) -> BaseDistribution:
        """
        Obtains the underlying link distribution.

        Returns:
            - link_model (distribution): The distribution instance.
        """
        return super().link_model()

    # ---------------------------------
    # ConditionalDistribution interface

    def mean(self, covariates: MatrixLike) -> Value:
        self.apply_regression(covariates)
        return self.link_model().mean()

    def variance(self, covariates: MatrixLike) -> Value:
        self.apply_regression(covariates)
        return self.link_model().variance()

    def log_prob(self, variate: VectorLike, covariates: MatrixLike) -> Value:
        self.apply_regression(covariates)
        return self.link_model().log_prob(variate)


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

    def regressor(self) -> RegressionDistribution:
        """
        Obtains a regression distribution for estimating
        parameters from variate and covariate data.

        Returns:
            - regressor (distribution): The regression distribution.
        """
        return self.regressor_klass(self.link_model())


###############################################################################
# Decorators for specifying the link model:


def add_link_model(
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
        _link_fn = klass.link_model

        def link_model(self) -> BaseDistribution:
            return link_klass(_link_fn(self))

        klass.link_model = link_model
        klass.link_model.__doc__ = _link_fn.__doc__
        return klass

    return decorator


def set_link_model(
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
        _link_fn = klass.link_model

        def link_model(self) -> BaseDistribution:
            return link_klass(self)

        klass.link_model = link_model
        klass.link_model.__doc__ = _link_fn.__doc__
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
