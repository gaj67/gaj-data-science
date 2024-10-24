"""
This module defines an 'interface' class for any algorithm that modifies
its behaviour based upon specialised parameters called 'controls'.

For convenience, common controls for iterative fitting (and their default values)
are also specified.
"""

from abc import ABC
from typing import TypeAlias, Dict, Any, Type, Callable


###############################################################################
# Base class and type:

# Encapsulates any and all settings necessary to control the algorithm.
# See Controllable.default_controls().
Controls: TypeAlias = Dict[str, Any]


class Controllable(ABC):
    """
    Encapsulates the typical controls and their default values.
    """

    @staticmethod
    def default_controls() -> Controls:
        """
        Provides default settings for controlling the algorithm.

        Parameters:
            - init (bool): Indicates whether or not to (re)initialise the
                parameter estimates; defaults to True. Set to False if the
                previous fit did not achieve convergence.
            - max_iters (int): The maximum number of iterations allowed.
            - step_iters (int): The maximum number of step-size line searches allowed.
            - score_tol (float): The minimum change in score to signal convergence.
            - param_tol (float): The minimum change in parameter values to signal convergence.
            - step_size (float): The parameter update scaling factor (or learning rate).
        """
        return {
            "init": True,
            "max_iters": 100,
            "step_iters": 10,
            "score_tol": 1e-8,
            "param_tol": 1e-6,
            "step_size": 1.0,
        }

    def get_controls(
        self,
        **controls: Controls,
    ) -> Controls:
        """
        Permits the default control values to be dynamically overridden.

        Input:
            - controls (dict): The user-specified controls. See default_controls().

        Returns:
            - results (dict): The summary output. See optimise_parameters().
        """
        # Obtain specified controls
        _controls = self.default_controls()
        _controls.update(controls)
        return _controls


###############################################################################
# Class decorator:


# Decorator for easily overriding the default values of controls
def set_controls(
    **controls: Controls,
) -> Callable[[Type[Controllable]], Type[Controllable]]:
    """
    Sstatically modifies the default values of the algorithm's controls.

    Input:
        - controls (dict): The overriden controls and their new default values.
            See Controllable.default_controls().

    Returns:
        - decorator (method): A decorator of a controllable class.
    """

    def decorator(klass: Type[Controllable]) -> Type[Controllable]:
        default_controls_fn = klass.default_controls

        @staticmethod
        def default_controls() -> Controls:
            _controls = default_controls_fn()
            _controls.update(controls)
            return _controls

        klass.default_controls = default_controls
        klass.default_controls.__doc__ = default_controls_fn.__doc__
        return klass

    return decorator
