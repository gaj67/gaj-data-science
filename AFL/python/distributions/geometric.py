"""
This module implements the geometric distribution.

The distributional parameter, theta, is the probability
of a stopping event. The variate, obs, is the total
number of trials, up to and including the stopping event.

The link parameter, eta = logit(theta), is non-canonical.
"""
# pylint: disable=wrong-import-position

import numpy as np

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
)

from .core.parameterised import UNSPECIFIED_VECTOR, guard_prob
from .core.distribution import StandardDistribution, set_link_model
from .core.link_models import LogitLink11
from .core.optimiser import Data, Controls


#################################################################
# Geometric distribution

DEFAULT_THETA = 0.5


@set_link_model(LogitLink11)
# pylint: disable=too-many-ancestors
class GeometricDistribution(StandardDistribution):
    """
    Implements the geometric probability distribution for a
    non-negative, integer, response variate, obs.

    The sole parameter, theta, governs the probability that obs=1.
    """

    def __init__(self, theta: Value = DEFAULT_THETA):
        """
        Initialises the geometric distribution(s).

        Input:
            - theta (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 1

    def default_parameters(self) -> Values:
        return (DEFAULT_THETA,)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        theta = params[0]
        return np.all(theta > 0) and np.all(theta < 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        theta = self.get_parameters()[0]
        return 1 / theta

    def variance(self) -> Value:
        theta = self.get_parameters()[0]
        return (1 - theta) / theta**2

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        theta = guard_prob(1.0 / data.mean())
        return (theta,)

    def compute_scores(self, variate: Vector) -> Vector:
        theta = self.get_parameters()[0]
        return np.log(theta) + (variate - 1) * np.log(1 - theta)

    def compute_gradients(self, variate: Vector) -> Values:
        theta = self.get_parameters()[0]
        g_t = 1 / theta - (variate - 1) / (1 - theta)
        return (g_t,)

    def compute_hessian(self, variate: Vector) -> Values2d:
        # Use expected value
        theta = self.get_parameters()[0]
        h_tt = -1 / (theta**2 * (1 - theta))
        return ((h_tt,),)


###############################################################################

if __name__ == "__main__":
    THETA = 0.123456

    def test1():
        """
        Test default parameter
        """
        gmd = GeometricDistribution()
        assert gmd.get_parameters() == (DEFAULT_THETA,)
        assert gmd.mean() == 1 / DEFAULT_THETA
        assert gmd.variance() == (1 - DEFAULT_THETA) / DEFAULT_THETA**2
        print("Passed default parameter tests!")

    def test2():
        """
        Test specified parameter
        """
        gmd = GeometricDistribution(THETA)
        assert gmd.get_parameters() == (THETA,)
        assert gmd.mean() == 1 / THETA
        assert gmd.variance() == (1 - THETA) / THETA**2
        print("Passed specified parameter tests!")

    def test3():
        """
        Test fitting 1 observation - a count of 0 is forbidden!
        """
        for obs in [1, 2, 5, 10, 100]:
            gmd = GeometricDistribution()
            res = gmd.fit(obs)
            assert res["converged"]
            assert np.abs(gmd.mean() - obs) < 1e-6
        print("Passed fitting 1 observation tests!")

    def test4():
        """
        Test fitting multiple observations
        """
        obs = [1, 2, 5, 10, 8, 13]
        gmd = GeometricDistribution()
        res = gmd.fit(obs)
        assert res["converged"]
        assert np.abs(gmd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting multiple observations tests!")

    def test5():
        """
        Test default regression parameters
        """
        gmr = GeometricDistribution().regressor()
        assert gmr.get_parameters() == (UNSPECIFIED_VECTOR,)
        print("Passed default regression parameter tests!")

    def test6():
        """
        Test fitting two groups of multiple observations
        """
        obs = [1, 2, 5, 10, 8, 13]
        covars0 = np.ones(6)
        covars1 = [-1, -1, -1, 1, 1, 1]
        covars = np.column_stack((covars0, covars1))
        gmr = GeometricDistribution().regressor()
        res = gmr.fit(obs, covars)
        assert res["converged"]
        assert np.abs(gmr.mean((1, -1)) - np.mean(obs[:3])) < 1e-6
        assert np.abs(gmr.mean((1, 1)) - np.mean(obs[3:])) < 1e-6
        print("Passed fitting grouped observations tests!")

    # ==========================================================
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
