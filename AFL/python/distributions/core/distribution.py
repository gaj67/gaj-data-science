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
    is_vector,
    is_divergent,
    to_matrix,
)

###############################################################################
# Handy methods for implementing distributions:


def guard_prob(value: Value) -> Value:
    """
    Guard against extreme probability values.

    Input:
        - value (float or vector): The value(s) to be checked.
    Returns:
        - value' (float or vector): The adjusted value(s).
    """
    if is_scalar(value):
        return 1e-30 if value <= 0.0 else 1 - 1e-10 if value >= 1.0 else value
    value = value.copy()
    value[value <= 0] = 1e-30
    value[value >= 1] = 1 - 1e-10
    return value


def guard_pos(value: Value) -> Value:
    """
    Guard against values going non-positive.

    Input:
        - value (float or vector): The value(s) to be checked.
    Returns:
        - value' (float or vector): The adjusted value(s).
    """
    if is_scalar(value):
        return 1e-30 if value <= 0.0 else value
    value = value.copy()
    value[value <= 0] = 1e-30
    return value


def as_scalar(value: Vector) -> Value:
    """
    If possible, converts a singleton vector into a scalar.

    Input:
        - value (vector): The vector value(s).

    Returns:
        - value' (flat or vector): The scalar or vector value.
    """
    if len(value) == 1:
        return value[0]
    return value


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
        print("DEBUG[Parameterised]: init")
        print("DEBUG: params=", params)
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
        self._params = params

    def is_valid_parameters(self, *params: Values) -> bool:
        """
        Determines whether or not the given parameter values are viable.
        Specifically, there should be the correct number of parameters, and
        each parameter should have finite value(s) within appropriate bounds.

        Input:
            - params (tuple of float or vector): The proposed parameter values.

        Returns:
            - flag (bool): A  value of True if the values are all valid,
                else False.
        """
        # By default, just check for dimensionality and divergence
        for value in params:
            if not is_scalar(value) and not is_vector(value):
                return False
            if is_divergent(value):
                return False
        return True


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
# Base regression classes:


class ConditionalDistribution(Parameterised):
    """
    A parameterised, conditional probability distribution of a scalar
    variate, X, given scalar or vector covariate(s), Z.

    It is always assumed that the regression parameters are bundled
    into the first parameter, which is transformed (using Z) into
    the link parameter of the underlying distribution.
    Any subsequent parameters are treated as independent parameters
    of the underlying distribution.

    Use an empty array if the number of regression weights is not
    known in advance of data fitting.
    """

    def __init__(self, reg_params: Vector, *indep_params: Values):
        """
        Initialises the conditional distribution.

        Input:
            - reg_params (vector): The value(s) of the regression parameter(s).
            - indep_params ((tuple of float, optional): The value(s) of
                the independent parameter(s), if any.
        """
        print("DEBUG[ConditionalDistribution]: init")
        print("DEBUG: phi=", reg_params)
        print("DEBUG: indep_params=", indep_params)
        super().__init__(reg_params, *indep_params)

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


class DelegatedDistribution(ConditionalDistribution):
    """
    Implements a conditional distribution by wrapping
    an unconditional distribution.
    """

    def __init__(self, pdf: Distribution, *params: Values):
        """
        Initialises the conditional distribution.

        Input:
            - pdf (distribution): The underlying distribution.
            - params (tuple of vector or floatt, optional): The
                parameter values of the conditional distribution.
        """
        print("DEBUG[DelegatedDistribution]: init")
        print("DEBUG: pdf=", pdf)
        print("DEBUG: params=", params)
        super().__init__(*params)
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

    def invert_regression(self, covariates: MatrixLike):
        """
        CComputes the regression function and then inverts the
        link parameter (and any independent parameters) into parameters
        of the underlying distribution.

        Input:
            - covariates (matrix-like): The covariate value(s).
        """
        phi, *psi = self.parameters()
        if len(phi) == 0:
            raise ValueError("Uninitialised regression parameters!")
        covs = to_matrix(covariates, n_cols=len(phi))
        eta = as_scalar(covs @ phi)
        print("DEBUG[invert_regression]: eta=", eta)
        params = self.inverse_link(eta, *psi)
        self.distribution().set_parameters(*params)

    @abstractmethod
    def inverse_link(self, *link_params: Values) -> Values:
        """
        Inverts the link function to map the link parameter and any independent
        parameters into the corresponding distributional parameters.

        Input:
            - link_params (tuple of float or vector): The conditional parameter values.

        Returns:
            - inv_params (tuple of float or vector): The distributional parameter values.
        """
        raise NotImplementedError
