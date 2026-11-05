"""
This module implements the beta-Bernoulli distribution.

The distributional parameters are the beta shape parameters,
namely alpha and beta.

The link parameter is chosen as eta = logit(mean) = log(alpha/beta),
which is not a natural parameter. The independent parameter is
chosen to be psi = alpha. 
"""

import numpy as np

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Vector,
)

from .core.distribution import StandardDistribution, set_link_model
from .core.link_models import LogRatioLink21b


#################################################################
# Beta-Bernoulli distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_link_model(LogRatioLink21b)
# pylint: disable=too-many-ancestors
class BetaBernoulliDistribution(StandardDistribution):
    """
    Implements the beta-Bernoulli probability distribution for a
    binary response variate, obs.

    The shape parameters, alpha and beta, govern the probability
    that obs=1.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the beta-Bernoulli distribution(s).

        Input:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 2

    def default_parameters(self) -> Values:
        return (DEFAULT_ALPHA, DEFAULT_BETA)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        alpha, beta = params
        return np.all(alpha > 0) and np.all(beta > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        alpha, beta = self.get_parameters()
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        theta = self.mean()
        return theta * (1 - theta)

    # -----------------------------
    # GradientOptimisable interface

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self.get_parameters()
        return (
            variate * np.log(alpha)
            + (1 - variate) * np.log(beta)
            - np.log(alpha + beta)
        )

    def compute_gradients(self, variate: Vector) -> Values:
        alpha, beta = self.get_parameters()
        g_a = variate / alpha - 1 / (alpha + beta)
        g_b = (1 - variate) / beta - 1 / (alpha + beta)
        return (g_a, g_b)


###############################################################################

if __name__ == "__main__":

    def test1():
        """
        Test default parameter
        """
        bbd = BetaBernoulliDistribution()
        assert len(bbd.get_parameters()) == 2
        assert bbd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
        mean = bbd.mean()
        assert mean == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
        assert bbd.variance() == mean * (1 - mean)
        print("Passed default parameter tests!")

    def test2():
        """
        Test fitting meanltiple observations
        """
        obs = np.array([1, 1, 0, 1])
        bbd = BetaBernoulliDistribution()
        res = bbd.fit(obs)
        assert res["converged"]
        assert np.abs(bbd.mean() - np.mean(obs)) < 1e-6
        # pylint: disable=unbalanced-tuple-unpacking
        _alpha, _beta = bbd.get_parameters()
        assert np.abs(_alpha / _beta - 3) < 1e-5
        print("Passed fitting meanltiple observations test!")

    def test3():
        """
        Test regression without intercept/bias
        """
        bbr = BetaBernoulliDistribution().regressor()
        obs = [1, 0]
        covars = [1, -1]
        res = bbr.fit(obs, covars, max_iters=1000)
        # NOTE: This is an ill-conditioned problem!
        assert not res["converged"]
        phi, _ = bbr.get_parameters()
        assert len(phi) == 1
        mean = bbr.mean(covars)
        for _obs, _mean in zip(obs, mean):
            assert np.abs(_mean - _obs) < 1e-2
        print("Passed ill-conditioned regression test!")

    def test4():
        """
        Test regression with intercept/bias
        """
        bbr = BetaBernoulliDistribution().regressor()
        obs = [1, 0, 1, 1, 0]
        covars = [(1, z) for z in [1, 1, -1, -1, -1]]
        res = bbr.fit(obs, covars)
        assert res["converged"]
        mean = bbr.mean((1, 1))
        assert np.abs(mean - 0.5) < 1e-5
        mean = bbr.mean([(1, -1)])
        assert np.abs(mean - 2 / 3) < 1e-5
        print("Passed fitting simple regression test!")

    ####################################################
    test1()
    test2()
    test3()
    test4()
