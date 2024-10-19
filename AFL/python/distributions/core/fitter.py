"""
This module defines the base classes for algorithms to fit observed data,
i.e. estimate parameters from data.

It is assumed that some parameterised objective function will be optimised. 
"""

from abc import abstractmethod
from typing import Optional, Type, Callable

import numpy as np

from .data_types import (
    VectorLike,
    to_vector,
)

from .distribution import Parameterised
from .optimiser import Optimiser, Data, Controls, Results


###############################################################################
# Abstract data fitting classes:

# NOTE: If Fittable subclasses ABC then @add_fitter() does NOT override
# abstract status!


class Fittable:
    """
    Provides an interface for estimating parameters from data.

    Assumes the underlying implementation subclasses Parameterised.
    """

    @abstractmethod
    def fit(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls,
    ) -> Results:
        """
        Estimates parameters from the given observation(s).

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - weights (vector-like, optional): The weight(s) of the data.
            - controls (dict): The user-specified controls.
                See Optimiser.default_controls().

        Returns:
            - results (dict): The summary output. See Optimiser.optimise_parameters().
        """
        raise NotImplementedError

    def to_data(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
    ) -> Data:
        """
        Bundles the observational data into standard format.

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - weights (vector-like, optional): The weight(s) of the data.

        Returns:
            - data (tuple of data): The bundled data.
        """
        # Allow for single or multiple observations, with or without weights
        v_data = to_vector(variate)
        n_rows = len(v_data)
        if weights is None:
            v_weights = np.ones(n_rows)
        else:
            v_weights = to_vector(weights, n_rows)
        return Data(v_data, v_weights)


###############################################################################
# Class decorators:


# Decorator for easily adding an optimiser implementation
def set_fitter(
    fitter_class: Type[Optimiser],
) -> Callable[[Type[Fittable]], Type[Fittable]]:
    """
    Implements the fit() method to wrap the distribution
    with an instance of the specified optimiser.

    Input:
        - fitter_class (class): The class of an optimiser implementation.

    Returns:
        - decorator (method): A decorator of a fittable and parameterised class.

    """

    def decorator(klass: Type[Fittable]) -> Type[Fittable]:

        if not (issubclass(klass, Parameterised) and issubclass(klass, Fittable)):
            raise ValueError("Class must be Parameterised & Fittable!")

        def fit(
            self,  # Parameterised & Fittable
            variate: VectorLike,
            weights: Optional[VectorLike] = None,
            **controls: Controls,
        ) -> Results:
            data = self.to_data(variate, weights)
            return fitter_class(self).fit(data, **controls)

        klass.fit = fit
        klass.fit.__doc__ = Fittable.fit.__doc__
        return klass

    return decorator
