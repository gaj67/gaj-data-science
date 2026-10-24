"""
This module implements the Poisson distribution.

The distributional parameter, lambda, is also the mean, mu.
The link parameter is the natural parameter, eta = log(lambda).
"""
# pylint: disable=wrong-import-position

import numpy as np
from scipy.special import gamma

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Data,
    Value,
    Values,
    Values2d,
    Vector,
)

from .core.controllable import Controls
from .core.parameterised import guard_pos
from .core.distribution import StandardDistribution, set_link_model
from .core.link_models import LogLink11


#################################################################
# Poisson distribution


DEFAULT_LAMBDA = 1.0


@set_link_model(LogLink11)
# pylint: disable=too-many-ancestors
class PoissonDistribution(StandardDistribution):
    """
    Implements the Poisson probability distribution for a binary
    response variate, obs.

    The sole parameter, lambda, governs the probability of
    obs = 0, 1, 2, ... .
    """

    def __init__(self, _lambda: Value = DEFAULT_LAMBDA):
        """
        Initialises the Poisson distribution(s).

        Input:
            - _lambda (float or ndarray): The distributional parameter value(s).
        """
        super().__init__(_lambda)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 1

    def default_parameters(self) -> Values:
        return (DEFAULT_LAMBDA,)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        _lambda = params[0]
        return np.all(_lambda > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        _lambda = self.get_parameters()[0]
        return _lambda

    def variance(self) -> Value:
        _lambda = self.get_parameters()[0]
        return _lambda

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        _lambda = guard_pos(data.mean())
        return (_lambda,)

    def compute_scores(self, variate: Vector) -> Vector:
        _lambda = self.get_parameters()[0]
        return variate * np.log(_lambda) - _lambda - np.log(gamma(variate + 1))

    def compute_gradients(self, variate: Vector) -> Values:
        _lambda = self.get_parameters()[0]
        g_l = (variate - _lambda) / _lambda
        return (g_l,)

    def compute_hessian(self, variate: Vector) -> Values2d:
        # Use expected value
        _lambda = self.get_parameters()[0]
        h_ll = -1 / _lambda
        return ((h_ll,),)


###############################################################################

if __name__ == "__main__":
    from .core.parameterised import UNSPECIFIED_VECTOR

    LAMBDA = 0.123456

    def test1():
        """
        Test default parameter
        """
        psd = PoissonDistribution()
        assert psd.get_parameters() == (DEFAULT_LAMBDA,)
        assert psd.mean() == DEFAULT_LAMBDA
        assert psd.variance() == DEFAULT_LAMBDA
        print("Passed default parameter tests!")

    def test2():
        """
        Test specified parameter
        """
        psd = PoissonDistribution(LAMBDA)
        assert psd.get_parameters() == (LAMBDA,)
        assert psd.mean() == LAMBDA
        assert psd.variance() == LAMBDA
        print("Passed specified parameter tests!")

    def test3():
        """
        Test fitting 1 observation - be careful of 0
        """
        for obs in [0, 1, 10, 100]:
            psd = PoissonDistribution()
            res = psd.fit(obs)
            assert res["converged"]
            assert np.abs(psd.mean() - obs) < 1e-6
        print("Passed fitting 1 observation tests!")

    def test4():
        """
        Test fitting multiple observations
        """
        for num_obs in range(2, 11):
            obs = np.random.randint(0, 100, num_obs)
            psd = PoissonDistribution()
            res = psd.fit(obs)
            assert res["converged"]
            assert np.abs(psd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting multiple observations tests!")

    def test5():
        """
        Test default parameter
        """
        psr = PoissonDistribution().regressor()
        assert psr.get_parameters() == (UNSPECIFIED_VECTOR,)
        print("Passed default regression parameter tests!")

    def test6():
        """
        Test fitting two groups of multiple observations
        """
        obs_p1 = [1, 2, 3, 1, 0, 0, 5, 4, 1, 0]
        covars_p1 = [1] * len(obs_p1)
        obs_m1 = np.array(obs_p1) / np.mean(obs_p1) ** 2
        covars_m1 = [-1] * len(obs_m1)
        obs = np.concatenate((obs_p1, obs_m1))
        covars = np.concatenate((covars_p1, covars_m1))
        psr = PoissonDistribution().regressor()
        res = psr.fit(obs, covars, score_tol=1e-12)
        assert res["converged"]
        mu_p1 = psr.mean(1)
        assert np.abs(mu_p1 - np.mean(obs_p1)) < 1e-6
        mu_m1 = psr.mean(-1)
        assert np.abs(mu_m1 - np.mean(obs_m1)) < 1e-6
        print("Passed fitting grouped observations tests!")

    def test7():
        """
        Test multiple regressions
        """
        obs_p1 = [1, 2, 3, 1, 0, 0, 5, 4, 1, 0]
        obs_m1 = np.array(obs_p1) / np.mean(obs_p1) ** 2
        obs = np.concatenate((obs_p1, obs_m1))
        covars_p1 = [(1, 1)] * len(obs_p1)
        covars_m1 = [(1, -1)] * len(obs_m1)
        covars = np.concatenate((covars_p1, covars_m1))
        psr = PoissonDistribution().regressor()
        res = psr.fit(obs, covars, score_tol=1e-12)
        assert res["converged"]
        phi0 = psr.get_parameters()[0][0]
        assert np.abs(phi0) < 1e-6
        mu_p1 = psr.mean([1, 1])
        assert np.abs(mu_p1 - np.mean(obs_p1)) < 1e-5
        mu_m1 = psr.mean([1, -1])
        assert np.abs(mu_m1 - np.mean(obs_m1)) < 1e-5
        print("Passed fitting multiple regression tests!")

    def test8():
        """
        Test regression with only bias
        """
        obs_p1 = [1, 2, 3, 1, 0, 0, 5, 4, 1, 0]
        obs_m1 = np.array(obs_p1) / np.mean(obs_p1) ** 2
        obs = np.concatenate((obs_p1, obs_m1))
        covars0 = [1] * len(obs)
        psr = PoissonDistribution().regressor()
        res = psr.fit(obs, covars0)
        assert res["converged"]
        weight = psr.get_parameters()[0][0]
        psd = PoissonDistribution()
        res = psd.fit(obs)
        assert res["converged"]
        eta0 = psd.get_parameters()[0]
        weight0 = np.log(eta0)
        assert np.abs(weight - weight0) < 1e-15
        eta1 = psr.link_model().invert_transform(weight)[0]
        assert np.abs(eta1 - eta0) < 1e-15
        weight1 = psr.link_model().apply_transform(eta1)[0]
        assert np.abs(weight - weight1) < 1e-15
        print("Passed bias-only regression tests!")

    # ==============================================================
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
