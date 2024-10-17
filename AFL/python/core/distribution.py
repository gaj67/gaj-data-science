"""
This module defines the base class for a probability distribution function (PDF)
of a (discrete or continuous) scalar variate, in terms of one or more
distributional parameters.

Each parameter may independently be specified as either a scalar value or
a vector of values (see the Value type). This also holds true for
the value(s) of the response variate.
"""

from abc import ABC, abstractmethod

from .data_types import (
    Value,
    Values,
    Vector,
    VectorLike,
    MatrixLike,
    is_scalar,
    is_divergent,
)

###############################################################################
# Base parameter classes:


class Parameterised(ABC):
    """
    Implements a mutable holder of parameters.
    """

    @staticmethod
    @abstractmethod
    def default_parameters() -> Values:
        """
        Provides default (scalar) values of the distributional parameters.

        Returns:
            - params (tuple of float): The default parameter values.
        """
        raise NotImplementedError

    def __init__(self, *params: Values):
        """
        Initialises the instance with the given parameter value(s).
        Each parameter may have either a single value or multiple values.

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        if len(params) == 0:
            self.set_parameters(*self.default_parameters())
        else:
            self.set_parameters(*params)

    def parameters(self) -> Values:
        """
        Provides the values of the distributional parameters.

        Returns:
            - params (tuple of float or ndarray): The parameter values.
        """
        return self._params

    def set_parameters(self, *params: Values):
        """
        Overrides the distributional parameter value(s).

        Invalid parameters will cause an exception.

        Input:
            - params (tuple of float or ndarray): The parameter value(s).
        """
        if not self.is_valid_parameters(*params):
            raise ValueError("Invalid parameters!")

        _params = []
        size = 1
        for i, param in enumerate(params):
            if not is_scalar(param):
                if len(param.shape) != 1:
                    raise ValueError(f"Expected parameter {i} to be uni-dimensional")
                _len = len(param)
                if _len <= 0:
                    raise ValueError(f"Expected parameter {i} to be non-empty")
                if _len == 1:
                    # Take scalar value
                    param = param[0]
                elif size == 1:
                    size = _len
                elif size != _len:
                    raise ValueError(f"Expected parameter {i} to have length {size}")
            _params.append(param)
        self._params = tuple(_params)
        self._size = size

    def is_valid_parameters(self, *params: Values) -> bool:
        """
        Determines whether or not the given parameter values are viable.
        Specifically, there should be the correct number of parameters, and
        each parameter should have finite value(s) within appropriate bounds.

        Input:
            - params (tuple of float): The proposed parameter values.

        Returns:
            - flag (bool): A  value of True if the values are valid,
                else False.
        """
        # By default, just check for  divergence
        for value in params:
            if is_divergent(value):
                return False
        return True


class Transformable:
    """
    Provides an invertable transformation between the default parameterisation
    and an alternative parameterisation.
    """

    def transform(self, *params: Values) -> Values:
        """
        Transforms the parameter values into an alternative representation.

        Input:
            - params (tuple of float-like or vector): The parameter values.

        Returns:
            - alt_params (tuple of float-like or vector): The alternative parameter values.
        """
        # By default, assume an identity transformation
        return params

    def inverse_transform(self, *alt_params: Values) -> Values:
        """
        Transforms the alternative parameter values into the usual representation.

        Input:
            - alt_params (tuple of float-like or vector): The alternative parameter values.

        Returns:
            - params (tuple of float-like or vector): The parameter values
        """
        # By default, assume an identity transformation
        return alt_params


###############################################################################
# Base distribution class:


class Distribution(Parameterised):
    """
    A parameterised probability distribution of a scalar variate, X.

    Each parameter may have either a single value or multiple values.
    If all parameters are single-valued, then only a single distribution
    is specified, and all computations, e.g. the distributional mean or
    variance, etc., will be single-valued.

    However, the use of one or more parameters with multiple values
    indicates a collection of distributions, rather than a single
    distribution. As such, all computations will be multi-valued
    rather than single-valued.
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
    def log_prob(self, data: VectorLike) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - data (vector-like): The value(s) of the response variate.

        Returns:
            - log_prob (float or vector): The log-likelihood(s).
        """
        raise NotImplementedError


###############################################################################
# Base regression class:


class ConditionalDistribution(Parameterised):
    """
    A parameterised, conditional probability distribution of a scalar
    variate, X, given scalar or vector covariate(s), Z.

    It is always assumed that the regression parameters are bundled
    into the first parameter, which is transformed (using Z) into
    the dependent parameter of the underlying distribution.
    Any subsequent parameters are treated as independent parameters
    of the underlying distribution.

    Use an empty array if the number of regression weights is not
    known in advance of data fitting.
    """

    def __init__(self, regression_params: Vector, independent_params: Vector):
        """
        Initialises the conditional distribution.

        Input:
            - regression_params (vector): The regression parameters.
            - independent_params (vector): The independent parameters.
        """
        super().__init__(regression_params, *independent_params)

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
    def log_prob(self, data: VectorLike, covariates: MatrixLike) -> Value:
        """
        Computes the log-likelihood(s) of the given data.

        Input:
            - data (vector-like): The value(s) of the response variate.
            - covariates (matrix-like): The covariate value(s).

        Returns:
            - log_prob (float or vector): The log-likelihood(s).
        """
        raise NotImplementedError
