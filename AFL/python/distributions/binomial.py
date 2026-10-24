"""
This module implements the binomial distribution for
a pre-specified number of Bernoulli trials.

The distributional parameters are n, the number of trials,
and theta, the probability of the event of interest.

The link parameter is the natural parameter, eta = logit(theta).
"""
# pylint: disable=wrong-import-position

import numpy as np

from scipy.special import loggamma  # pylint: disable=no-name-in-module


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

from .core.parameterised import guard_prob
from .core.distribution import StandardDistribution, set_link_model, add_link_model
from .core.link_models import SwapLink20, LogitLink21
from .core.controllable import Controls


#################################################################
# Binomial distribution

DEFAULT_TRIALS = 1
DEFAULT_THETA = 0.5


@add_link_model(LogitLink21)  # eta = logit(theta) <-> theta = logistic(eta)
@set_link_model(SwapLink20)  # swap order of (n, theta) <-> (theta, n)
# pylint: disable=too-many-ancestors
class BinomialDistribution(StandardDistribution):
    """
    Implements the binomial probability distribution for a non-negative
    response variate, obs.

    The first parameter, n, governs the number of Bernoulli trials.
    The second parameter, theta, governs the probability that any
    independent trial is a special event.
    """

    def __init__(self, num_trials: int, theta: Value = DEFAULT_THETA):
        """
        Initialises the binomial distribution(s).

        Input:
            - num_trials (int): The pre-determined number of trials.
            - theta (float or ndarray): The probability value(s).
        """
        super().__init__(num_trials, theta)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 2

    def default_parameters(self) -> Values:
        return (DEFAULT_TRIALS, DEFAULT_THETA)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        num_trials, theta = params
        return num_trials > 0 and np.all(theta > 0) and np.all(theta < 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        num_trials, theta = self.get_parameters()
        return num_trials * theta

    def variance(self) -> Value:
        num_trials, theta = self.get_parameters()
        return num_trials * theta * (1 - theta)

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        num_trials = self.get_parameters()[0]
        theta = guard_prob(data.mean() / num_trials)
        return (num_trials, theta)

    def compute_scores(self, variate: Vector) -> Vector:
        num_trials, theta = self.get_parameters()
        return (
            loggamma(num_trials + 1)
            - loggamma(variate + 1)
            - loggamma(num_trials - variate + 1)
            + variate * np.log(theta)
            + (num_trials - variate) * np.log(1 - theta)
        )

    def compute_gradients(self, variate: Vector) -> Values:
        num_trials, theta = self.get_parameters()
        g_t = (variate - num_trials * theta) / (theta * (1 - theta))
        return (0, g_t)

    def compute_hessian(self, variate: Vector) -> Values2d:
        # Force to be invertible, since n is constant
        num_trials, theta = self.get_parameters()
        h_tt = -num_trials / (theta * (1 - theta))
        return ((1, 0), (0, h_tt))


###############################################################################

if __name__ == "__main__":
    from .core.parameterised import UNSPECIFIED_VECTOR

    NUM_TRIALS = 5
    THETA = 0.123456

    def test1():
        """
        Test default parameter
        """
        bnd = BinomialDistribution(NUM_TRIALS)
        assert bnd.get_parameters() == (NUM_TRIALS, DEFAULT_THETA)
        assert bnd.default_parameters() == (DEFAULT_TRIALS, DEFAULT_THETA)
        assert bnd.mean() == NUM_TRIALS * DEFAULT_THETA
        assert bnd.variance() == NUM_TRIALS * DEFAULT_THETA * (1 - DEFAULT_THETA)
        print("Passed default parameter tests!")

    def test2():
        """
        Test specified parameter
        """
        bnd = BinomialDistribution(1, THETA)
        assert bnd.get_parameters() == (1, THETA)
        assert bnd.mean() == THETA
        assert bnd.variance() == THETA * (1 - THETA)
        print("Passed specified parameter tests!")

    def test3():
        """
        Test fitting 1 observation - be careful with 0 or n!
        """
        for obs in range(NUM_TRIALS + 1):
            bnd = BinomialDistribution(NUM_TRIALS)
            res = bnd.fit(obs)
            assert res["converged"]
            assert np.abs(bnd.mean() - obs) < 1e-6
        print("Passed fitting 1 observation tests!")

    def test4():
        """
        Test fitting multiple observations
        """
        obs = [1, 1, 0, 5, 3, 2, 4, 4, 1, 0]
        bnd = BinomialDistribution(NUM_TRIALS, THETA)
        res = bnd.fit(obs)
        assert res["converged"]
        assert np.abs(bnd.mean() - np.mean(obs)) < 1e-6

        for num_trials in range(2, 11):
            obs = np.random.randint(0, NUM_TRIALS + 1, num_trials)
            bnd = BinomialDistribution(NUM_TRIALS)
            res = bnd.fit(obs)
            assert res["converged"]
            assert np.abs(bnd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting multiple observations tests!")

    def test5():
        """
        Test default parameter
        """
        bnr = BinomialDistribution(NUM_TRIALS).regressor()
        assert bnr.get_parameters() == (UNSPECIFIED_VECTOR, NUM_TRIALS)
        print("Passed default regression parameter tests!")

    def test6():
        """
        Test fitting two groups of multiple observations
        """
        obs_m1 = [1, 2, 3, 1, 0, 0, 1, 2, 1, 0]
        covar_m1 = [-1] * len(obs_m1)
        obs_p1 = [NUM_TRIALS - x for x in obs_m1]
        covar_p1 = [1] * len(obs_p1)
        obs = obs_m1 + obs_p1
        covar = covar_m1 + covar_p1
        bnr = BinomialDistribution(NUM_TRIALS).regressor()
        res = bnr.fit(obs, covar, score_tol=1e-12)
        assert res["converged"]
        mu_p1 = bnr.mean(1)
        assert np.abs(mu_p1 - np.mean(obs_p1)) < 1e-6
        mu_m1 = bnr.mean(-1)
        assert np.abs(mu_m1 - np.mean(obs_m1)) < 1e-6
        print("Passed fitting grouped observations tests!")

    # =========================================================
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
