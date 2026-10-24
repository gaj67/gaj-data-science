"""
This modules defines classes for estimating parameters from data
(without covariates) using an optimiser.
"""

from typing import Optional

from .data_types import VectorLike, to_data
from .controllable import Controls, Controllable, set_controls
from .optimisable import Optimisable
from .optimiser import Results, Optimiser, NonlinearOptimiser, LeastSquaresOptimiser


###############################################################################
# Main class for optimisation:


@set_controls(
    # (bool) Replaces standard optimisation with nonlinear optimisation
    nonlinear=False,
    # (bool) Replaces standard or nonlinear optimisation with least-squares fitting
    least_squares=False,
)
class Fittable(Optimisable, Controllable):
    """
    Interface for estimating parameter values from data.
    """

    def fit(
        self,
        variate: VectorLike,
        weights: Optional[VectorLike] = None,
        **controls: Controls
    ) -> Results:
        """
        Estimates optimal parameter values from the data.

        Inputs:
            - variate (vector-like): The value(s) of the variate.
            - weights (vector-like, optional): The weight(s) of the data.
            - controls (dict): The user-specified controls.
                See Controllable.default_controls().

        Returns:
            - res (dict): The summary of the estimation algorithm.
        """
        _data = to_data(variate, weights)
        _controls = self.get_controls(**controls)
        if _controls.get("least_squares", False):
            _optimiser = LeastSquaresOptimiser(self)
        elif _controls.get("nonlinear", False):
            _optimiser = NonlinearOptimiser(self)
        else:
            _optimiser = Optimiser(self)
        res = _optimiser.fit(_data, _controls)
        return res
