"""
This module implements the beta-geometric distribution.

The variate X is the total number of Bernoulli trials,
up to and including the stopping event.

The distributional parameters are the beta shape parameters,
namely alpha and beta. Note that the mean of X does not exist
if alpha <= 1, and no variance exists if alpha <= 2.

Implicitly, we model theta ~ Beta(alpha, beta),
where theta is the probability of halting a geometric
sequence of Bernoulli trials. Thus, the distribution

  X ~ BetaGeom(alpha, beta)
  
is equivalent to the hierarchical model

  X | theta ~ Geom(theta)
  theta | alpha, beta ~ Beta(alpha, beta)

Note that this can also be used to model a run of successes,
where X is the total length of the run before failure,
conditioned on having observed the first success.
"""
# pylint: disable=wrong-import-position

import numpy as np

# pylint: disable=no-name-in-module
from scipy.special import (
    digamma,
    betaln as logbeta,
)

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
# Beta-Bernoulli distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_controls(nonlinear=True)
# pylint: disable=too-many-ancestors
class BetaGeometricDistribution(StandardDistribution):
    """
    Implements the beta-geometric probability distribution for a
    response variate, X.

    The shape parameters, alpha and beta, govern the probability
    that X=1, 2, ..., infinity.

    The prior distribution, theta ~ Beta(alpha,  beta), controls
    the probability of halting the sequence of trials.
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
        return (alpha + beta - 1) / (alpha - 1) if alpha > 1 else np.inf

    def variance(self) -> Value:
        alpha, beta = self._get_parameters2()
        return (
            alpha * beta * (alpha + beta - 1) / ((alpha - 1) ** 2 * (alpha - 2))
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
        beta = (mu_x - 1) * (alpha - 1)
        if beta <= 0:
            return self.default_parameters()
        return (alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self._get_parameters2()
        scores = logbeta(alpha + 1, beta + variate - 1) - logbeta(alpha, beta)
        return scores

    # -----------------------------
    # GradientOptimisable interface

    def compute_gradients(self, variate: Vector) -> Values:
        # NOTE: The expanded log-likelihood is:
        # scores = (
        #    np.log(alpha) + loggamma(alpha + beta)
        #    + loggamma(beta + variate - 1)
        #    - loggamma(beta)
        #    - loggamma(alpha + beta + variate)
        # )
        alpha, beta = self._get_parameters2()
        g_a = 1 / alpha + digamma(alpha + beta) - digamma(alpha + beta + variate)
        g_b = (
            digamma(alpha + beta)
            + digamma(beta + variate - 1)
            - digamma(beta)
            - digamma(alpha + beta + variate)
        )
        if self._is_symmetric:
            return (g_a + g_b,)
        return (g_a, g_b)


###############################################################################

if __name__ == "__main__":
    import scipy.stats as s_stats

    def test1():
        """
        Test default parameters.
        """
        bgd = BetaGeometricDistribution()
        assert len(bgd.get_parameters()) == 2
        assert bgd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
        bgd = BetaGeometricDistribution(symmetric=True)
        assert len(bgd.get_parameters()) == 1
        assert bgd.get_parameters() == (DEFAULT_ALPHA,)
        print("Passed default parameter tests!")

    def test2():
        """
        Test fitting multiple observations.
        """
        obs = [1, 2, 5, 1, 3, 1]
        bgd = BetaGeometricDistribution()
        res = bgd.fit(obs)
        assert res["converged"]
        bgd = BetaGeometricDistribution(symmetric=True)
        res = bgd.fit(obs)
        assert res["converged"]
        print("Passed fitting multiple observations tests!")

    def test3():
        """
        Test recovery of hierarchical model.
        """
        alpha0 = 3
        beta0 = 5
        num_samples = 2000
        thetas = s_stats.beta.rvs(alpha0, beta0, size=num_samples, random_state=42)
        obs = s_stats.geom.rvs(thetas, size=num_samples, random_state=42)

        bgd = BetaGeometricDistribution()
        res = bgd.fit(obs)
        assert res["converged"]
        alpha, beta = bgd.get_parameters()  # pylint: disable=unbalanced-tuple-unpacking
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
