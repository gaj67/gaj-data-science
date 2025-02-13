"""
This module implements the beta-binomial distribution.

The distributional parameters are the number, n, of 
Bernoulli trials, and the beta shape parameters,
alpha and beta.

The link parameters are chosen as eta1 = log(beta) and
eta2 = log(alpha). The independent parameter is
chosen to be psi = n. 
"""

import numpy as np
from scipy.special import loggamma, digamma, polygamma


if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Vector,
)

from .core.distribution import StandardDistribution, set_link_model, add_link_model
from .core.link_models import ReverseLinkn0, LogLinkn2
from .core.optimiser import Data, Controls
from .core.data_types import Values, Values2d


#################################################################
# Beta-binomial distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


def log_choose(n: Value, m: Value) -> Value:
    return loggamma(n + 1) - loggamma(m + 1) - loggamma(n - m + 1)


@add_link_model(LogLinkn2)  # (eta1, eta2, psi) = (log(beta), log(alpha), n)
@set_link_model(ReverseLinkn0)  # (n, alpha, beta) <-> (beta, alpha, n)
class BetaBinomialDistribution(StandardDistribution):
    """
    Implements the binomial probability distribution for a non-negative
    response variate, X.

    The first parameter, n, governs the number of independent Bernoulli
    trials. The remaining parameters, alpha and beta, govern the
    probability that any given trial is a 'success'.
    """

    def __init__(self, n: int, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the beta-binomial distribution(s).

        Input:
            - n (int): The number of trials.
            - alpha (float or ndarray): The prior number of 'successes'.
            - beta (float or ndarray): The prior number of 'failures'.
        """
        super().__init__(n, alpha, beta)

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (0, DEFAULT_ALPHA, DEFAULT_BETA)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 3 or not super().check_parameters(*params):
            return False
        n, alpha, beta = params
        return (n > 0) and np.all(alpha > 0) and np.all(beta > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        n, alpha, beta = self.get_parameters()
        return n * alpha / (alpha + beta)

    def variance(self) -> Value:
        n, alpha, beta = self.get_parameters()
        mu = alpha / (alpha + beta)
        sigsq = mu * (1 - mu) / (alpha + beta + 1)
        return n * mu * (1 - mu) + n * (n -  1) * sigsq

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimate(self, data: Data, controls: Controls) -> Values:
        n = self.get_parameters()[0]
        # TODO Provide proper estimates
        alpha = DEFAULT_ALPHA
        beta = DEFAULT_BETA
        return (n, alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        n, alpha, beta = self.get_parameters()
        # Use negative hyper-geometric form
        s = variate  # successes
        f = n - variate  # failures
        return (
            log_choose(alpha + s - 1, s)
            + log_choose(beta + f - 1, f)
            - log_choose(alpha + beta + n - 1, n)
        )

    def compute_gradients(self, variate: Vector) -> Values:
        n, alpha, beta = self.get_parameters()
        s = variate  # successes
        f = n - variate  # failures
        t0 = digamma(alpha + beta) - digamma(alpha + beta + n)
        g_a = t0 + digamma(alpha + s) - digamma(alpha)
        g_b = t0 + digamma(beta + f) - digamma(beta)
        return (0, g_a, g_b)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        n, alpha, beta = self.get_parameters()
        # Use mean field approximation to expectation of Hessian
        s_ab = alpha + beta
        s_abn = s_ab + n
        nh_ab =  polygamma(1, s_abn) - polygamma(1, s_ab)
        nh_aa = nh_ab + polygamma(1, alpha) - polygamma(1, alpha * s_abn / s_ab)
        nh_bb = nh_ab + polygamma(1, beta) - polygamma(1, beta * s_abn / s_ab)
        # Force to be invertible, since n is constant
        return ((1, 0, 0), (0, nh_aa, nh_ab), (0, nh_ab, nh_bb))
