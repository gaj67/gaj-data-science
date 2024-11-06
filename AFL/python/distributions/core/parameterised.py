"""
This module defines the base class for an object that has one or more
parameters.

Each parameter may independently be specified as either a scalar value or
a vector of values (see the data_types.Value type).
"""

from abc import ABC, abstractmethod

import numpy as np

from .data_types import (
    Value,
    Values,
    is_scalar,
    is_vector,
    is_divergent,
)

###############################################################################
# Handy methods for guarding against impossible parameter values:


ESPSILON0 = 1e-10
ESPSILON1 = 1 - 1e-10


def guard_prob(value: Value) -> Value:
    """
    Guard against extreme probability values.

    Input:
        - value (float or vector): The value(s) to be checked.
    Returns:
        - value' (float or vector): The adjusted value(s).
    """
    if is_scalar(value):
        return ESPSILON0 if value <= 0.0 else ESPSILON1 if value >= 1.0 else value
    value = np.array(value, dtype=float)
    value[value <= 0] = ESPSILON0
    value[value >= 1] = ESPSILON1
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
        return ESPSILON0 if value <= 0.0 else value
    value = np.array(value, dtype=float)
    value[value <= 0] = ESPSILON0
    return value


###############################################################################
# Base parameter class:


class Parameterised(ABC):
    """
    Interface for a mutable holder of parameters.
    """

    @abstractmethod
    def default_parameters(self) -> Values:
        """
        Provides default values of the distributional parameters.

        Returns:
            - params (tuple of scalar or vector): The default parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Values:
        """
        Provides the values of the distributional parameters.

        Returns:
            - params (tuple of scalar or vector): The parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, *params: Values):
        """
        Overrides the distributional parameter value(s).

        Invalid parameters will cause an exception.

        Input:
            - params (tuple of scalar or vector): The parameter value(s).
        """
        raise NotImplementedError

    def check_parameters(self, *params: Values) -> bool:
        """
        Determines whether or not the given parameter values are viable.
        Specifically, there should be the correct number of parameters, and
        each parameter should have finite value(s) within appropriate bounds.

        Input:
            - params (tuple of scalar or vector): The proposed parameter values.

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
# Base implementation class:


class Parameters(Parameterised):
    """
    Partial implementation of a holder of parameter values.

    Each parameter may have either a single (scalar) value
    or multiple (vector) values.
    """

    def __init__(self, *params: Values):
        """
        Initialises the instance with the given parameter value(s).

        Input:
            - params (tuple of scalar or vector): The parameter value(s).
        """
        if len(params) == 0:
            self.set_parameters(*self.default_parameters())
        else:
            self.set_parameters(*params)

    def get_parameters(self) -> Values:
        return self._params

    def set_parameters(self, *params: Values):
        if not self.check_parameters(*params):
            print("DEBUG[Parameters.set_parameters]: params=", params)
            raise ValueError("Invalid parameters!")
        self._params = params


###############################################################################
# Interface for parameter transformation:


class TransformParameterised(Parameterised):
    """
    Interface for an invertible transformation between a standard parameter
    space and an alternative parameter space.
    """

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        std_params = self.underlying().default_parameters()
        return self.apply_transform(*std_params)

    def get_parameters(self) -> Values:
        std_params = self.underlying().get_parameters()
        return self.apply_transform(*std_params)

    def set_parameters(self, *params: Values):
        if not self.check_parameters(*params):
            raise ValueError("Invalid parameters!")
        std_params = self.invert_transform(*params)
        self.underlying().set_parameters(*std_params)

    # ------------------------------
    # TransformParameterised interface

    @abstractmethod
    def underlying(self) -> Parameterised:
        """
        Obtains the underlying parameterised model.

        Returns:
            - inst (parameterised): The underlying instance.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_transform(self, *std_params: Values) -> Values:
        """
        Transforms the standard parameterisation into the
        alternative parameterisation.

        Input:
            - std_params (tuple of scalar or vector): The value(s)
                of the underlying parameter(s).

        Returns:
            - alt_params (tuple of scalar or vector): The value(s)
                of the transformed parameter(s).
        """
        raise NotImplementedError

    @abstractmethod
    def invert_transform(self, *alt_params: Values) -> Values:
        """
        Inversely transforms the alternative parameterisation into
        the standard parameterisation.

        Input:
            - alt_params (tuple of scalar or vector): The value(s)
                of the transformed parameter(s).

        Returns:
            - std_params (tuple of scalar or vector): The value(s)
                of the underlying parameter(s).
        """
        raise NotImplementedError


###############################################################################
# Interface for regression:


# The default value of the link parameter
DEFAULT_LINK = 0.0


class RegressionParameterised(Parameterised):
    """
    An interface for accessing the regression parameters and
    associated independent parameters of a regression model.

    It is assumed that there are two underlying models.
    The regression model contains the regression parameters.
    The link model contains the link parameter as first parameter,
    followed by any independent parameters.

    The link parameter, eta, and regression parameters, phi,
    are assumed to be directly related by a regression function
    of the covariates, Z, namely: eta = f(Z, phi).
    Although this function is not used explicitly here, we do
    make use of the assumption that f(Z, 0) = 0.
    """

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        phi = self.regression().default_parameters()[0]
        psi = self.link().default_parameters()[1:]
        return (phi, *psi)

    def get_parameters(self) -> Values:
        phi = self.regression().get_parameters()[0]
        psi = self.link().get_parameters()[1:]
        return (phi, *psi)

    def set_parameters(self, *params: Values):
        if not self.check_parameters(*params):
            print("DEBUG[RegressionParameterised.set_parameters]: params=", params)
            raise ValueError("Invalid parameters!")
        phi, *psi = params
        self.regression().set_parameters(phi)
        self.link().set_parameters(DEFAULT_LINK, *psi)

    def check_parameters(self, *params: Values) -> bool:
        # Must have at least one parameter!
        if len(params) == 0:
            return False
        phi, *psi = params
        if not self.regression().check_parameters(phi):
            return False
        return self.link().check_parameters(DEFAULT_LINK, *psi)

    # ---------------------------------
    # RegressionParameterised interface

    @abstractmethod
    def regression(self) -> Parameterised:
        """
        Obtains the underlying regression model.

        Returns:
            - regression (parameterised): The regression model.
        """
        raise NotImplementedError

    @abstractmethod
    def link(self) -> Parameterised:
        """
        Obtains the underlying link model.

        Returns:
            - link (parameterised): The link model.
        """
        raise NotImplementedError


###############################################################################
# Base class for regression:


# Indicates that the vector size is not yet specified
UNSPECIFIED_VECTOR = np.array([])


class OneVectorParameters(Parameters):
    """
    Implements a holder of a single vector of parameters.
    """

    def default_parameters(self):
        return (UNSPECIFIED_VECTOR,)

    def check_parameters(self, *params):
        if len(params) != 1:
            return False
        phi = params[0]
        return is_vector(phi) and not is_divergent(phi)
