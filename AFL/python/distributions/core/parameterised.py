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
# Base implementation classes:


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
            raise ValueError("Invalid parameters!")
        self._params = params


# Indicates that the vector size is not yet specified
UNSPECIFIED_VECTOR = np.array([])


class VectorParameters(Parameters):
    """
    Implements a holder of a vector parameters.
    """

    def __init__(self, num_params: int):
        """
        Initialises space for the specified number
        of vector parameters.
        
        Input:
            - num_params (int): The required number
                of vector parameters.
        """
        if num_params < 0:
            raise ValueError("Number of parameters must be non-negative!")
        self._num_params = num_params
        Parameters.__init__(self)
        
    def default_parameters(self) -> Values:
        return tuple([UNSPECIFIED_VECTOR] * self._num_params)

    def check_parameters(self, *params) -> bool:
        if len(params) != self._num_params:
            return False
        for v in params:
            if not is_vector(v) or is_divergent(v):
                return False
        return True


###############################################################################
# Interface for parameter transformation:


class TransformParameterised(Parameterised):
    """
    A partial implementation of an invertible transformation between
    a standard parameter space and an alternative parameter space.
    """

    # ------------------------------
    # TransformParameterised interface

    def __init__(self, underlying: Parameterised):
        """
        Initialises the transormation with the underlying parameters.
        
        Input:
            - underlying (parameterised): The underlying parameters
                of the standard parameter space.
        """
        self._underlying = underlying

    def underlying(self) -> Parameterised:
        """
        Obtains the underlying parameterised model.

        Returns:
            - inst (parameterised): The underlying instance.
        """
        return self._underlying

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


###############################################################################
# Interface for regression:


# The default value of the link parameter
DEFAULT_LINK = 0.0


class RegressionParameters(Parameterised):
    """
    An implementation for accessing regression parameters and
    associated independent parameters.

    It is assumed that there are two underlying models.
    The regression model contains the regression parameters.
    The link model contains the link parameters followed by
    any independent parameters.

    Each scalar link parameter, eta, and corresponding
    vector regression parameter, phi, are assumed to be
    directly related by a regression function of the
    covariates, Z, namely: eta = f(Z, phi).
    Although this function is not used explicitly here, we do
    make use of the assumption that f(Z, 0) = 0.
    """

    # ------------------------------
    # RegressionParameters interface

    def __init__(self, num_links: int, link_model: Parameterised):
        """
        Initialises the regression model parameters.
        
        Input:
            - num_links (int): The required number
                of regression parameters.
            - link_model (parameterised): The underlying link model.
        """
        self._num_links = num_links
        self._regression_parameters = VectorParameters(num_links)
        self._link_model = link_model

    def regression_model(self) -> Parameterised:
        """
        Obtains the underlying regression model.

        Returns:
            - regression (parameterised): The regression model.
        """
        return self._regression_parameters

    def link_model(self) -> Parameterised:
        """
        Obtains the underlying link model.

        Returns:
            - link_model (parameterised): The link model.
        """
        return self._link_model

    def num_links(self) -> int:
        """
        Obtains the number of regressable link parameters of
        the underlying link model.

        Returns:
            - num_links (int): The number of link parameters.
        """
        return self._num_links

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        phis = self.regression_model().default_parameters()
        psi = self.link_model().default_parameters()[self.num_links():]
        return (*phis, *psi)

    def get_parameters(self) -> Values:
        phis = self.regression_model().get_parameters()
        psi = self.link_model().get_parameters()[self.num_links():]
        return (*phis, *psi)

    def set_parameters(self, *params: Values):
        if not self.check_parameters(*params):
            raise ValueError("Invalid parameters!")
        phis = params[0:self._num_links]
        psi = params[self._num_links:]
        self.regression_model().set_parameters(*phis)
        etas = [DEFAULT_LINK] * self.num_links()
        self.link_model().set_parameters(*etas, *psi)

    def check_parameters(self, *params: Values) -> bool:
        phis = params[0:self.num_links()]
        psi = params[self.num_links():]
        if not self.regression_model().check_parameters(*phis):
            return False
        etas = [DEFAULT_LINK] * self.num_links()
        return self.link_model().check_parameters(*etas, *psi)
