"""
This module implements the beta-Bernoulli distribution
for completed runs, i.e. terminated, contiguous sequences
of binary trials of the same outcome. For example,
it might model a run of successes terminated by a failure.

The probability of terminating a run is assumed to be
sampled from a beta distribution prior to each run.
The distributional parameters are the beta shape parameters,
namely alpha and beta.

The variate X denotes the length of the run, not including the
terminating event. Note that probability of a run is
conditioned on having observed the first event of the run,
e.g. the first success in a run of successes.
Thus, X must be positive integer.
"""

import numpy as np

# pylint: disable=no-name-in-module
from scipy.special import (
    digamma,
    betaln as logbeta,
)

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Vector,
    Data,
)

from .core.distribution import StandardDistribution
from .core.controllable import Controls, set_controls

#################################################################
# Beta-Bernoulli runs distribution


# Assume a uniform prior. Note: The mean and variance will not exist!
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_controls(nonlinear=True)
# pylint: disable=too-many-ancestors
class BetaBernoulliRunsDistribution(StandardDistribution):
    """
    Implements the beta-Bernoulli runs distribution for a
    a positive integer variate, X, which is the length of
    the run.

    The shape parameters, alpha and beta, govern the probability
    that a run will be terminated.
    """

    _is_symmetric = False

    def __init__(
        self,
        alpha: Value = DEFAULT_ALPHA,
        beta: Value = DEFAULT_BETA,
        symmetric: bool = False,
    ):
        """
        Initialises the beta-geometric distribution(s).

        Input:
            - alpha (float or ndarray): Optional first shape parameter value(s).
            - beta (float or ndarray): Optional second shape parameter value(s).
            - symmetric (bool): Optional flag indicating whether beta = alpha (if True) or not.
        """
        super().__init__()
        if symmetric:
            self._is_symmetric = True
            self.set_parameters(alpha)
        else:
            self.set_parameters(alpha, beta)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 1 if self._is_symmetric else 2

    def default_parameters(self) -> Values:
        if self._is_symmetric:
            return (DEFAULT_ALPHA,)
        return (DEFAULT_ALPHA, DEFAULT_BETA)

    def check_parameters(self, *params: Values) -> bool:
        if not super().check_parameters(*params):
            return False
        alpha = params[0]
        beta = alpha if self._is_symmetric else params[1]
        return np.all(alpha > 0) and np.all(beta > 0)

    def _get_parameters2(self) -> Values:
        params = self.get_parameters()
        if self._is_symmetric:
            params = (params[0], params[0])
        return params

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        alpha, beta = self._get_parameters2()
        return (alpha + beta) / (alpha - 1) if alpha > 1 else np.inf

    def variance(self) -> Value:
        alpha, beta = self._get_parameters2()
        return (
            alpha * (beta + 1) * (alpha + beta) / ((alpha - 1) ** 2 * (alpha - 2))
            if alpha > 2
            else np.inf
        )

    # ---------------------
    # Optimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        # Use method of moments
        mu_x = data.mean()
        var_x = data.variance()
        # Estimate alpha
        alpha = 2 * var_x / (var_x - mu_x * (mu_x - 1))
        if alpha <= 0:
            return self.default_parameters()
        if self._is_symmetric:
            return (alpha,)
        # Estimate  beta
        beta = (mu_x - 1) * (alpha - 1) - 1
        if beta <= 0:
            return self.default_parameters()
        return (alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self._get_parameters2()
        scores = logbeta(alpha + 1, beta + variate) - logbeta(alpha, beta + 1)
        return scores

    # -----------------------------
    # GradientOptimisable interface

    def compute_gradients(self, variate: Vector) -> Values:
        # NOTE: The expanded log-likelihood is:
        # scores = (
        #    np.log(alpha) + loggamma(alpha + beta + 1)
        #    + loggamma(beta + variate)
        #    - loggamma(beta + 1)
        #    - loggamma(alpha + beta + variate + 1)
        # )
        alpha, beta = self._get_parameters2()
        g_a = (
            1 / alpha + digamma(alpha + beta + 1) - digamma(alpha + beta + variate + 1)
        )
        g_b = (
            digamma(alpha + beta + 1)
            + digamma(beta + variate)
            - digamma(beta + 1)
            - digamma(alpha + beta + variate + 1)
        )
        if self._is_symmetric:
            return (g_a + g_b,)
        return (g_a, g_b)


###############################################################################

if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from scipy.special import beta as b_fn
    import scipy.stats as s_stats

    def test1():
        """
        Test default parameters.
        """
        brd = BetaBernoulliRunsDistribution()
        assert len(brd.get_parameters()) == 2
        assert brd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
        brd = BetaBernoulliRunsDistribution(symmetric=True)
        assert len(brd.get_parameters()) == 1
        assert brd.get_parameters() == (DEFAULT_ALPHA,)
        print("Passed default parameter tests!")

    def test2():
        """
        Test mean and variance.
        """
        alpha0 = 5
        beta0 = 3
        brd = BetaBernoulliRunsDistribution(alpha0, beta0)
        mu_x = brd.mean()
        mu_theory = b_fn(alpha0 - 1, beta0 + 1) / b_fn(alpha0, beta0 + 1)
        assert np.abs(mu_x - mu_theory) < 1e-6
        var_x = brd.variance()
        var_theory = (
            b_fn(alpha0 - 2, beta0 + 1) / b_fn(alpha0, beta0 + 1)
            - mu_theory**2
            + b_fn(alpha0 - 2, beta0 + 2) / b_fn(alpha0, beta0 + 1)
        )
        assert np.abs(var_x - var_theory) < 1e-6

    def test3():
        """
        Test recovery of hierarchical model.
        """
        alpha0 = 3
        beta0 = 4
        num_samples = 2000
        thetas = s_stats.beta.rvs(alpha0, beta0 + 1, size=num_samples, random_state=42)
        obs = s_stats.geom.rvs(thetas, size=num_samples, random_state=42)

        brd = BetaBernoulliRunsDistribution()
        res = brd.fit(obs)
        assert res["converged"]
        alpha, beta = brd.get_parameters()  # pylint: disable=unbalanced-tuple-unpacking
        p_tol = max(np.abs(alpha - alpha0), np.abs(beta - beta0))
        print("Geround-truth:", alpha0, beta0)
        print("Estimates:", alpha, beta)
        print("MAD[params] =", p_tol)
        print("Passed hierarchical modelling tests!")

    # ======================================================================
    # Perform testing
    test1()
    test2()
    test3()
