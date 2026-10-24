"""
This module implements the Bernoulli distribution.

The distributional parameter, theta0, is also the mean, mu.
The link parameter is the natural parameter, eta = logit(theta0).
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
# Bernoulli distribution

DEFAULT_THETA = 0.5


# pylint: disable=too-many-ancestors


@set_link_model(LogitLink11)
class BernoulliDistribution(StandardDistribution):
    """
    Implements the Bernoulli probability distribution for a binary
    response variate, obs.

    The sole parameter, theta0, governs the probability that obs=1.
    """

    def __init__(self, theta0: Value = DEFAULT_THETA):
        """
        Initialises the Bernoulli distribution(s).

        Input:
            - theta0 (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(theta0)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 1

    def default_parameters(self) -> Values:
        return (DEFAULT_THETA,)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        theta0 = params[0]
        return np.all(theta0 > 0) and np.all(theta0 < 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        theta0 = self.get_parameters()[0]
        return theta0

    def variance(self) -> Value:
        theta0 = self.get_parameters()[0]
        return theta0 * (1 - theta0)

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        theta0 = guard_prob(data.mean())
        return (theta0,)

    def compute_scores(self, variate: Vector) -> Vector:
        theta0 = self.get_parameters()[0]
        return variate * np.log(theta0) + (1 - variate) * np.log(1 - theta0)

    def compute_gradients(self, variate: Vector) -> Values:
        theta0 = self.get_parameters()[0]
        grads = (variate - theta0) / (theta0 * (1 - theta0))
        return (grads,)

    def compute_hessian(self, variate: Vector) -> Values2d:
        # Use expected value
        theta0 = self.get_parameters()[0]
        neg_var = -1.0 / (theta0 * (1 - theta0))
        return ((neg_var,),)


###############################################################################

if __name__ == "__main__":
    THETA = 0.123456

    def test1():
        """
        Test default parameter
        """
        bnd = BernoulliDistribution()
        assert bnd.get_parameters() == (DEFAULT_THETA,)
        assert bnd.mean() == DEFAULT_THETA
        assert bnd.variance() == DEFAULT_THETA * (1 - DEFAULT_THETA)
        print("Passed default parameter tests!")

    def test2():
        """
        Test specified parameter
        """
        bnd = BernoulliDistribution(THETA)
        assert bnd.get_parameters() == (THETA,)
        assert bnd.mean() == THETA
        assert bnd.variance() == THETA * (1 - THETA)
        print("Passed specified parameter tests!")

    def test3():
        """
        Test fitting 1 observation - be careful with 0 or 1!
        """
        for obs in [1e-3, 0.9, 0.5, 0.1, 1, 0]:
            bnd = BernoulliDistribution()
            res = bnd.fit(obs)
            assert res["converged"]
            assert np.abs(bnd.mean() - obs) < 1e-6
        print("Passed fitting 1 observation tests!")

    def test4():
        """
        Test fitting multiple observations
        """
        obs = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
        bnd = BernoulliDistribution(THETA)
        res = bnd.fit(obs)
        assert res["converged"]
        assert np.abs(bnd.mean() - np.mean(obs)) < 1e-6

        for num_obs in range(2, 11):
            obs = np.random.randint(0, 2, num_obs)
            bnd = BernoulliDistribution()
            res = bnd.fit(obs)
            assert res["converged"]
            assert np.abs(bnd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting multiple observations tests!")

    def test5():
        """
        Test default regression parameter
        """
        bnr = BernoulliDistribution().regressor()
        assert bnr.get_parameters() == (UNSPECIFIED_VECTOR,)
        print("Passed default regression parameter tests!")

    def test6():
        """
        Test fitting two groups of multiple observations
        """
        obs_p1 = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
        covars_p1 = [1] * len(obs_p1)
        obs_m1 = [1 - x for x in obs_p1]
        covars_m1 = [-1] * len(obs_m1)
        obs = obs_p1 + obs_m1
        covars = covars_p1 + covars_m1
        bnr = BernoulliDistribution().regressor()
        res = bnr.fit(obs, covars, score_tol=1e-12)
        assert res["converged"]
        mu_p1 = bnr.mean(1)
        assert np.abs(mu_p1 - np.mean(obs_p1)) < 1e-6
        mu_m1 = bnr.mean(-1)
        assert np.abs(mu_m1 - np.mean(obs_m1)) < 1e-6
        print("Passed fitting grouped observations tests!")

    def test7():
        """
        Test regression with only bias
        """
        obs_p1 = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
        covars_p1 = [1] * len(obs_p1)
        bnr = BernoulliDistribution().regressor()
        res = bnr.fit(obs_p1, covars_p1)
        assert res["converged"]
        wp1 = bnr.get_parameters()[0][0]
        bnd = BernoulliDistribution()
        res = bnd.fit(obs_p1)
        assert res["converged"]
        theta0 = bnd.get_parameters()[0]
        weight0 = np.log(theta0 / (1 - theta0))
        assert np.abs(wp1 - weight0) < 1e-15
        theta1 = bnr.link_model().invert_transform(wp1)[0]
        assert np.abs(theta1 - theta0) < 1e-15
        weight1 = bnr.link_model().apply_transform(theta1)[0]
        assert np.abs(wp1 - weight1) < 1e-15
        print("Passed bias-only regression tests!")

    # ===========================================================
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
