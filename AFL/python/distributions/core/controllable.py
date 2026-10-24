"""
This modules defines classes, types and methods for controlling
model fitting and parameter optimisation.
"""

from typing import TypeAlias, Callable, Type, Dict, Any


###############################################################################
# Control mechanisms:

# Encapsulates input control settings
Controls: TypeAlias = Dict[str, Any]


# Permits control of an iterative optimiser or fitter
class Controllable:
    """
    Encapsulates the typical controls and their default values.
    """

    _controls: Controls = {}

    @classmethod
    def default_controls(cls) -> Controls:
        """
        Provides default control settings for the class.
        """
        return cls._controls

    def get_controls(
        self,
        **controls: Controls,
    ) -> Controls:
        """
        Dynamically obtains and optionally (but temporarily) updates the default controls.

        Input:
            - controls (dict): The user-specified controls.

        Returns:
            - _controls (dict): The composite controls.
        """
        # Obtain specified controls
        _controls = self.default_controls().copy()
        _controls.update(controls)
        return _controls


# Decorator for easily overriding the default values of controls
def set_controls(
    **controls: Controls,
) -> Callable[[Type[Controllable]], Type[Controllable]]:
    """
    Statically modifies the default controls.

    Input:
        - controls (dict): The overriden controls and their new default values.
            See Controllable.default_controls().

    Returns:
        - decorator (method): A decorator of a controllable class.
    """

    def decorator(cls: Type[Controllable]) -> Type[Controllable]:
        _controls = cls.default_controls().copy()
        _controls.update(controls)
        cls._controls = _controls  # pylint: disable=protected-access
        return cls

    return decorator
