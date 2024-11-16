"""
This module implements the beta distribution.

The distributional parameters are alpha (shape) and beta (shape).
The link parameter is taken to be eta = log(alpha/beta) = logit(mu),
which is not a natural parameter. The independent parameter is
choseen as psi = alpha.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import digamma, polygamma

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
from .core.distribution import StandardDistribution, set_link
from .core.link_models import LogRatioLink2a


#################################################################
# Beta distribution and fitter


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_link(LogRatioLink2a)
class BetaDistribution(StandardDistribution):
    """
    Implements the beta probability distribution for a proportional
    response variate, X.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the beta distribution(s).

        Input:
            - alpha (float or ndarray): The first shape parameter value(s).
            - beta (float or ndarray): The second shape parameter value(s).
        """
        super().__init__(alpha, beta)

    # -----------------------
    # Parameterised interface

    def default_parameters(self) -> Values:
        return (DEFAULT_ALPHA, DEFAULT_BETA)

    def check_parameters(self, *params: Values) -> bool:
        if len(params) != 2 or not super().check_parameters(*params):
            return False
        alpha, beta = params
        return np.all(alpha > 0) and np.all(beta > 0)

    # ----------------------
    # Distribution interface

    def mean(self) -> Value:
        alpha, beta = self.get_parameters()
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        alpha, beta = self.get_parameters()
        nu = alpha + beta
        mu = alpha / nu
        return mu * (1 - mu) / (nu + 1)

    # -----------------------------
    # GradientOptimisable interface

    def compute_estimates(self, variate: Vector) -> Values:
        # Backoff estimate based only on mean
        x = guard_prob(variate)
        beta = DEFAULT_BETA
        alpha = beta * x / (1 - x)
        ind = np.ones(len(variate), dtype=bool)
        return ind, (alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self.get_parameters()
        return beta_dist.logpdf(variate, alpha, beta)

    def compute_gradients(self, variate: Vector) -> Values:
        # grad = (dL/d alpha, dL/d beta)
        alpha, beta = self.get_parameters()
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        return (np.log(variate) - mu_alpha, np.log(1 - variate) - mu_beta)

    def compute_neg_hessian(self, variate: Vector) -> Values2d:
        alpha, beta = self.get_parameters()
        cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
        v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
        v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
        return ((v_alpha, cov_ab), (cov_ab, v_beta))


###############################################################################


if __name__ == "__main__":
    # Test default parameter
    bd = BetaDistribution()
    assert bd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
    _mu = bd.mean()
    assert _mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
    assert bd.variance() == _mu * (1 - _mu) / (1 + DEFAULT_ALPHA + DEFAULT_BETA)
    print("Passed default parameter tests!")

    # Test fitting two observations
    X = np.array([0.25, 0.75])
    bd = BetaDistribution()
    res = bd.fit(X)
    _mu = bd.mean()
    assert np.abs(_mu - np.mean(X)) < 1e-6
    _alpha, _beta = bd.get_parameters()
    assert np.abs(_alpha - _beta) < 1e-6
    print("Passed simple fitting test!")

    # Test same data via regression - means are complementary
    br = BetaDistribution().regressor()
    Z = [-1, 1]
    try:
        res = br.fit(X, Z)
    except ValueError:
        print("Two sample regression failed to converge!")
        res = br.fit(X, Z, max_iters=0)
    for _x, _z in zip(X, Z):
        _mu = br.mean(_z)
        assert np.abs(_mu - _x) < 1e-6
    print("Passed divergent regression test!")

    # Test regression on two groups - means are NOT complementary
    Xm1 = [0.1, 0.2, 0.3]  # mean 0.2
    Xp1 = [0.6, 0.7, 0.8]  # mean 0.7
    Zm1 = [(1, -1)] * len(Xm1)
    Zp1 = [(1, 1)] * len(Xp1)
    X = Xm1 + Xp1
    Z = Zm1 + Zp1
    br = BetaDistribution().regressor()
    res = br.fit(X, Z)
    assert res["converged"]
    mum1 = br.mean(Zm1[0])
    assert np.abs(mum1 - np.mean(Xm1)) < 1e-2
    mup1 = br.mean(Zp1[0])
    assert np.abs(mup1 - np.mean(Xp1)) < 1e-1
    print("Passed two-group regression test!")
