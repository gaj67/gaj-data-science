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
psi = alpha, which is almost but not quite a natural parameter.
"""

import numpy as np
from scipy.special import loggamma, digamma, polygamma

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
)

from .core.parameterised import guard_prob
from .core.distribution import StandardDistribution, add_link_model, set_link_model
from .core.link_models import SwapLink20, LogitLink21
from .core.optimiser import set_controls


#################################################################
# Beta distribution and fitter


# Default to geometric distribution
DEFAULT_THETA = 0.5
DEFAULT_ALPHA = 1


@add_link_model(LogitLink21)  # logit theta = f(Z)
@set_link_model(SwapLink20)   # Swap order of alpha and theta
@set_controls(max_iters=1000)
class NegBinomialDistribution(StandardDistribution):
    """
    Implements the negative binomial probability distribution
    for an integer (count) response variate, X.
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

    def default_parameters(self) -> Values:
        return (DEFAULT_ALPHA, DEFAULT_THETA)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
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
        ind = variate > 0
        alpha = DEFAULT_ALPHA
        theta = alpha / (alpha + variate[ind])
        return ind, (alpha, theta)

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
        # grad = (dL/d alpha, dL/d theta)
        alpha, theta = self.get_parameters()
        y_theta = -variate / (1 - theta)
        mu_theta = -alpha / theta  # E[Y_theta]
        y_alpha = digamma(alpha + variate)
        mu_alpha = digamma(alpha) - np.log(theta)  # E[Y_alpha]
        return (y_alpha - mu_alpha, y_theta - mu_theta)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        # hess = [(d^L/d alpha^2, d^L/d alpha.d theta),
        #         (d^L/d theta.d alpha, d^L/d theta^2)]
        alpha, theta = self.get_parameters()
        v_theta = alpha / (theta**2 * (1 - theta))  # Var[Y_theta]
        # Use mean field approximation
        v_alpha = polygamma(1, alpha) - polygamma(1, alpha / theta)  # Var[Y_alpha]
        cov_ta = -1 / theta  # Cov[Y_theta, Y_alpha]
        return ((v_alpha, cov_ta), (cov_ta, v_theta))


###############################################################################


if __name__ == "__main__":
    # Test default parameter
    nd = NegBinomialDistribution()
    assert nd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_THETA)
    print("Passed default parameter tests!")

    # Test fitting multiple observations
    X = [0, 1, 1, 2, 5]
    nd = NegBinomialDistribution()
    res = nd.fit(X)
    assert res["converged"]
    assert np.abs(nd.mean() - np.mean(X)) < 1e-6
    print("Passed simple fitting tests!")

    # Test fitting multiple observations
    for n in range(5, 101):
        while True:
            X = np.random.randint(0, 20, n)
            if np.var(X) > np.mean(X):
                break
        nd = NegBinomialDistribution()
        res = nd.fit(X)
        assert res["converged"]
        assert np.abs(nd.mean() - np.mean(X)) < 1e-6
    print("Passed fitting multiple observations tests!")

    # Test regression of two groups
    X = [0, 1, 1, 2, 5]
    Z = [-1, -1, -1, 1, 1]
    nr = NegBinomialDistribution().regressor()
    res = nr.fit(X, Z)
    assert res["converged"]
    assert nr.mean(1) > nr.mean(-1)
    print("Passed two-group regression tests!")

    # Test regression with only bias
    Z0 = [1] * len(X)
    nr = NegBinomialDistribution().regressor()
    res = nr.fit(X, Z0)
    wvec, a = nr.get_parameters()
    w = wvec[0]
    nd = NegBinomialDistribution()
    res = nd.fit(X)
    a0, t0 = nd.get_parameters()
    w0 = np.log(t0 / (1 - t0))
    assert np.abs(w - w0) < 1e-9
    assert np.abs(a - a0) < 1e-9
    t1, a1 = nr.link_model().invert_transform(w, a)
    assert np.abs(t1 - t0) < 1e-9
    assert np.abs(a1 - a0) < 1e-9
    w2, a2 = nr.link_model().apply_transform(t1, a1)
    assert np.abs(w - w2) < 1e-9
    assert np.abs(a - a2) < 1e-9
    print("Passed bias-only regression tests!")
