"""
This module implements the negative binomial distribution.

The distributional parameters are theta (probability) and
alpha (number). The theta parameter (sometimes labelled as
'p') gives the probability that a single Bernoulli trial
results in a special event (e.g. a 'success' or, alternatively,
a 'failure'). The alpha parameter specifies the required
number of (not necessarily consecutive) special events in a
sequence of independent Bernoulli trials, at which point the
trials cease. 

Note: alpha being integer-valued gives the Pascal distribution,
whereas alpha being real-valued gives the Polya distriibution.

The link parameter is taken to be eta = logit(theta), which is
not a natural parameter. The independent parameter is chosen as
psi = alpha, which is almost (but not quite) a natural parameter.
"""
# pylint: disable=wrong-import-position

import numpy as np

# pylint: disable=no-name-in-module
from scipy.special import (
    loggamma,
    digamma,
    polygamma,
)

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
)

from .core.distribution import StandardDistribution, add_link_model, set_link_model
from .core.link_models import SwapLink20, LogitLink21
from .core.optimiser import set_controls


#################################################################
# Beta distribution and fitter


# Default to geometric distribution
DEFAULT_THETA = 0.5
DEFAULT_ALPHA = 1


@add_link_model(LogitLink21)  # logit theta = f(covars)
@set_link_model(SwapLink20)  # Swap order of alpha and theta
@set_controls(max_iters=1000)
# pylint: disable=too-many-ancestors
class NegBinomialDistribution(StandardDistribution):
    """
    Implements the negative binomial probability distribution
    for an integer (count) response variate, obs.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, theta: Value = DEFAULT_THETA):
        """
        Initialises the negative binomial distribution(s).

        Input:
            - alpha (scalar or vector): The required event count value(s).
            - theta (scalar or vector): The event probability value(s).
        """
        super().__init__(alpha, theta)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 2

    def default_parameters(self) -> Values:
        return (DEFAULT_ALPHA, DEFAULT_THETA)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        alpha, theta = params
        return np.all(alpha > 0) and np.all(theta > 0) and np.all(theta < 1)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        alpha, theta = self.get_parameters()
        return alpha * (1 - theta) / theta

    def variance(self) -> Value:
        alpha, theta = self.get_parameters()
        return alpha * (1 - theta) / theta**2

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimates(self, variate: Vector) -> Values:
        # Backoff estimate based only on mean
        alpha = DEFAULT_ALPHA
        theta = np.ones(len(variate))
        ind = variate > 0
        theta[ind] = alpha / (alpha + variate[ind])
        theta[~ind] = np.nan
        return (alpha, theta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, theta = self.get_parameters()
        return (
            loggamma(alpha + variate)
            - loggamma(alpha)
            - loggamma(variate + 1)
            + variate * np.log(1 - theta)
            + alpha * np.log(theta)
        )

    def compute_gradients(self, variate: Vector) -> Values:
        alpha, theta = self.get_parameters()
        g_a = digamma(alpha + variate) - digamma(alpha) + np.log(theta)
        g_t = -variate / (1 - theta) + alpha / theta
        return (g_a, g_t)

    def compute_hessian(self, variate: Vector) -> Values2d:
        # Use expected values with mean field approximation
        alpha, theta = self.get_parameters()
        h_aa = polygamma(1, alpha / theta) - polygamma(1, alpha)
        h_tt = -alpha / (theta**2 * (1 - theta))
        h_at = 1 / theta
        return ((h_aa, h_at), (h_at, h_tt))


###############################################################################


if __name__ == "__main__":

    def test1():
        """
        Test default parameter
        """
        nbd = NegBinomialDistribution()
        assert nbd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_THETA)
        print("Passed default parameter tests!")

    def test2():
        """
        Test fitting multiple observations
        """
        obs = [0, 1, 1, 2, 5]
        nbd = NegBinomialDistribution()
        res = nbd.fit(obs)
        assert res["converged"]
        assert np.abs(nbd.mean() - np.mean(obs)) < 1e-6
        print("Passed simple fitting tests!")

    def test3():
        """
        Test fitting multiple observations
        """
        for num_trials in range(5, 101):
            while True:
                obs = np.random.randint(0, 20, num_trials)
                if np.var(obs) > np.mean(obs):
                    break
            nbd = NegBinomialDistribution()
            res = nbd.fit(obs)
            assert res["converged"]
            assert np.abs(nbd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting multiple observations tests!")

    def test4():
        """
        Test regression of two groups
        """
        obs = [0, 1, 1, 2, 5]
        covars = [-1, -1, -1, 1, 1]
        nbr = NegBinomialDistribution().regressor()
        res = nbr.fit(obs, covars)
        assert res["converged"]
        assert nbr.mean(1) > nbr.mean(-1)
        print("Passed two-group regression tests!")

    def test5():
        """
        Test regression with only bias
        """
        obs = [0, 1, 1, 2, 5]
        covars0 = [1] * len(obs)
        nbr = NegBinomialDistribution().regressor()
        res = nbr.fit(obs, covars0)
        assert res["converged"]
        wvec, alpha = nbr.get_parameters()
        weight = wvec[0]
        nbd = NegBinomialDistribution()
        res = nbd.fit(obs)
        assert res["converged"]
        (
            alpha0,
            theta0,
        ) = nbd.get_parameters()  # pylint: disable=unbalanced-tuple-unpacking
        weight0 = np.log(theta0 / (1 - theta0))
        assert np.abs(weight - weight0) < 1e-6
        assert np.abs(alpha - alpha0) < 1e-6
        theta1, alpha1 = nbr.link_model().invert_transform(weight, alpha)
        assert np.abs(theta1 - theta0) < 1e-6
        assert np.abs(alpha1 - alpha0) < 1e-6
        weight2, alpha2 = nbr.link_model().apply_transform(theta1, alpha1)
        assert np.abs(weight - weight2) < 1e-6
        assert np.abs(alpha - alpha2) < 1e-6
        print("Passed bias-only regression tests!")

    # =================================================
    test1()
    test2()
    test3()
    test4()
    test5()
