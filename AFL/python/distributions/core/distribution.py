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

from .parameterised import Parameterised, Parameters
from .fittable import Scorable, Differentiable, Fittable
from .regressable import Linkable, Regressable


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


class StandardDistribution(Parameters, Distribution, Fittable):
    """
    Implements a parameterised distribution with an optimisable
    log-likelihood score.
    """

    # Distribution interface

    def log_prob(self, variate: VectorLike) -> Value:
        return as_value(self.compute_scores(to_vector(variate)))


###############################################################################
# Link distribution class - this may sit between a StandardDistribution and a
# RegressionDistribution.


class LinkDistribution(Linkable, Distribution):
    """
    Encapsulates a distribution using an alternative parameterisation
    to the standard form.

    There exists an invertible transformation between the standard
    parameterisation of the underlying distribution and the alternative
    parameterisation.
    """

    def __init__(self, dist: Distribution, *params: Values):
        """
        Encapsulates an instance of the specified distribution, and initialises
        the values of the alternative parameters.

        Input:
            - dist (distribution): An instance of a parameterised distribution.
            - params (tuple of float or vector): The value(s) of the alternative
                parameter(s).
        """
        for klass in (Parameterised, Distribution, Scorable, Differentiable):
            if not isinstance(dist, klass):
                raise NotImplementedError(f"Missing {klass.__name__} interface!")
        self._dist = dist
        if len(params) == 0:
            params = self.default_parameters()
        self.set_parameters(*params)

    def underlying(self) -> Distribution:
        """
        Obtains the underlying distribution.

        Returns:
            - dist (distribution): The distribution instance.
        """
        return self._dist

    # Parameterised interface

    def set_parameters(self, *params: Values):
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")
        std_params = self.invert_link(*params)
        self.underlying().set_parameters(*std_params)

    def parameters(self) -> Values:
        std_params = self.underlying().parameters()
        return self.apply_link(*std_params)

    # Distribution interface

    def mean(self) -> Value:
        return self.underlying().mean()

    def variance(self) -> Value:
        return self.underlying().variance()

    def log_prob(self, variate: VectorLike) -> Value:
        return self.underlying().log_prob(variate)

    # Scorable interface

    def compute_scores(self, variate: Vector) -> Vector:
        return self.underlying().compute_scores(variate)

    # Differentiable interface

    def compute_gradients(self, variate: Vector) -> Values:
        jac = self.compute_jacobian()
        grad = self.underlying().compute_gradients(variate)
        return mult_rmat_vec(jac, grad)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        n_hess = self.underlying().compute_neg_hessian(variate)
        if len(n_hess) == 0:
            return n_hess
        jac = self.compute_jacobian()
        # This is an approximation which assumes that
        # the expectation of any score gradient is zero.
        mat = mult_rmat_rmat(jac, n_hess)
        return mult_rmat_rmat(mat, jac)

    # LinkDistribution interface

    @abstractmethod
    def apply_link(self, *std_params: Values) -> Values:
        """
        Applies the link function to transform the standard
        parameterisation into the alternative parameterisation.

        Input:
            - std_params (tuple of float or vector): The value(s)
                of the standard parameter(s).

        Returns:
            - link_params (tuple of float or vector): The value(s)
                of the alternative parameter(s).
        """
        raise NotImplementedError

    @abstractmethod
    def invert_link(self, *link_params: Values) -> Values:
        """
        Applies the inverse link function to transform the alternative
        parameterisation into the standard parameterisation.

        Input:
            - link_params (tuple of float or vector): The value(s)
                of the alternative parameter(s).

        Returns:
            - std_params (tuple of float or vector): The value(s)
                of the standard parameter(s).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_jacobian(self) -> Values2d:
        """
        Computes the derivatives of the inverse link function, i.e. the
        standard parameters (column-wise), with respect to the alternative
        parameters (row-wise).

        Returns:
            - jac (matrix-like of scalar or vector): The Jacobian matrix
                of the inverse link transformation.
        """
        raise NotImplementedError


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


class RegressionDistribution(Parameters, ConditionalDistribution, Regressable):
    """
    Implements a conditional distribution using linear regression to compute
    the parameters of the underlying distribution.
    """

    def __init__(self, dist: Distribution, *params: Values):
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
        Parameters.__init__(self, *params)
        if not isinstance(dist, Distribution):
            raise NotImplementedError("Missing link Distribution interface!")
        Regressable.__init__(self, dist)

    def underlying(self) -> Distribution:
        """
        Obtains the underlying link distribution.

        Returns:
            - dist (distribution): The distribution instance.
        """
        return Regressable.underlying(self)

    def regression_parameters(self) -> Vector:
        """
        Obtains the regression parameters.

        Returns:
            - reg_params (vector): The value(s) of the regression parameter(s).
        """
        return self.parameters()[0]

    def independent_parameters(self) -> Values:
        """
        Obtains the independent parameters.

        Returns:
            - indep_params (tuple of float or vector): The value(s) of the independent
                parameter(s), if any.
        """
        return self.parameters()[1:]

    # Parameterised interface

    def is_valid_parameters(self, *params: Values) -> bool:
        # Must have a vector first parameter!
        if len(params) == 0:
            return False
        _iter = iter(params)
        phi = next(_iter)
        if not is_vector(phi) or is_divergent(phi):
            return False
        # Any remaining parameters must be scalar or vector
        for value in _iter:
            if not is_scalar(value) and not is_vector(value):
                return False
            if is_divergent(value):
                return False
        return True

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
