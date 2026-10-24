"""
This module implements the beta-binomial distribution.

The distributional parameters are the number, n, of 
Bernoulli trials, and the beta shape parameters,
alpha and beta.

The link parameters are chosen as eta1 = log(beta) and
eta2 = log(alpha). The independent parameter is
chosen to be psi = n. 
"""
# pylint: disable=wrong-import-position

import numpy as np
from scipy.special import loggamma, digamma  # pylint: disable=no-name-in-module


if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Data,
    Value,
    Values,
    Vector,
)

from .core.distribution import StandardDistribution, set_link_model, add_link_model
from .core.link_models import ReverseLinkn0, LogLinkn2
from .core.controllable import Controls


#################################################################
# Beta-binomial distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


def log_choose(num: Value, denom: Value) -> Value:
    """
    Generalised logarithm of the from-N choose-M function.
    """
    return loggamma(num + 1) - loggamma(denom + 1) - loggamma(num - denom + 1)


@add_link_model(LogLinkn2)  # (eta1, eta2, psi) = (log(beta), log(alpha), n)
@set_link_model(ReverseLinkn0)  # (n, alpha, beta) <-> (beta, alpha, n)
# pylint: disable=too-many-ancestors
class BetaBinomialDistribution(StandardDistribution):
    """
    Implements the binomial probability distribution for a non-negative
    response variate, X.

    The first parameter, n, governs the number of independent Bernoulli
    trials. The remaining parameters, alpha and beta, govern the
    probability that any given trial is a 'success'.
    """

    def __init__(
        self, num_trials: int, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA
    ):
        """
        Initialises the beta-binomial distribution(s).

        Input:
            - num_trials (int): The fixed number of trials.
            - alpha (float or ndarray): The prior number of 'successes'.
            - beta (float or ndarray): The prior number of 'failures'.
        """
        super().__init__(num_trials, alpha, beta)

    # -----------------------
    # Parameterised interface

    def num_parameters(self) -> int:
        return 2

    def default_parameters(self) -> Values:
        return (0, DEFAULT_ALPHA, DEFAULT_BETA)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 3 or not super().check_parameters(*params):
            return False
        num_trials, alpha, beta = params
        return (num_trials > 0) and np.all(alpha > 0) and np.all(beta > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        num_trials, alpha, beta = self.get_parameters()
        return num_trials * alpha / (alpha + beta)

    def variance(self) -> Value:
        num_trials, alpha, beta = self.get_parameters()
        mean = alpha / (alpha + beta)
        sigsq = mean * (1 - mean) / (alpha + beta + 1)
        return num_trials * mean * (1 - mean) + num_trials * (num_trials - 1) * sigsq

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        num_trials = self.get_parameters()[0]
        alpha = DEFAULT_ALPHA
        beta = DEFAULT_BETA
        return (num_trials, alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        num_trials, alpha, beta = self.get_parameters()
        # Use negative hyper-geometric form
        return (
            log_choose(alpha + variate - 1, variate)
            + log_choose(beta + num_trials - variate - 1, num_trials - variate)
            - log_choose(alpha + beta + num_trials - 1, num_trials)
        )

    def compute_gradients(self, variate: Vector) -> Values:
        num_trials, alpha, beta = self.get_parameters()
        term0 = digamma(alpha + beta) - digamma(alpha + beta + num_trials)
        g_a = term0 + digamma(alpha + variate) - digamma(alpha)
        g_b = term0 + digamma(beta + num_trials - variate) - digamma(beta)
        return (0, g_a, g_b)
