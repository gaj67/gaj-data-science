"""
This module implements the beta distribution.

The distributional parameters are alpha (shape) and beta (shape),
which are both given log links to the regression function.

As an optional variation, a symmetric version of the beta distribution
takes only the alpha parameter, and forces beta = alpha.
"""
# pylint: disable=wrong-import-position

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import digamma, polygamma  # pylint: disable=no-name-in-module

if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import Value, Values, Values2d, Vector, Data

from .core.controllable import Controls
from .core.parameterised import guard_prob
from .core.distribution import StandardDistribution, set_link_model
from .core.link_models import LogLink22


#################################################################
# Beta distribution and fitter


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


# pylint: disable=too-many-ancestors


@set_link_model(LogLink22)
class BetaDistribution(StandardDistribution):
    """
    Implements the beta probability distribution for a proportional
    response variate, obs.
    """

    _is_symmetric = False

    def __init__(
        self,
        alpha: Value = DEFAULT_ALPHA,
        beta: Value = DEFAULT_BETA,
        symmetric: bool = False,
    ):
        """
        Initialises the beta distribution(s).

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
        return alpha / (alpha + beta)

    def variance(self) -> Value:
        alpha, beta = self._get_parameters2()
        nu_p = alpha + beta
        mu_p = alpha / nu_p
        return mu_p * (1 - mu_p) / (nu_p + 1)

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        # Use approximate method of moments
        mu_x = guard_prob(data.mean())
        var_x = data.variance()
        nu_est = var_x / (mu_x * (1 - mu_x))
        alpha = nu_est * mu_x
        beta = nu_est * (1 - mu_x)
        if alpha <= 0 or beta <= 0:
            return self.default_parameters()
        if self._is_symmetric:
            alpha = 0.5 * (alpha + beta)
            return (alpha,)
        return (alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self._get_parameters2()
        return beta_dist.logpdf(variate, alpha, beta)

    def compute_gradients(self, variate: Vector) -> Values:
        # L = (
        #   loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta)
        #   + (alpha - 1) * np.log(variate) + (beta - 1) * np.log(1 - variate)
        # )
        alpha, beta = self._get_parameters2()
        thetas = guard_prob(variate)
        d_nu = digamma(alpha + beta)
        g_a = d_nu - digamma(alpha) + np.log(thetas)
        g_b = d_nu - digamma(beta) + np.log(1 - thetas)
        if self._is_symmetric:
            return (g_a + g_b,)
        return (g_a, g_b)

    def compute_hessian(self, variate: Vector) -> Values2d:
        alpha, beta = self._get_parameters2()
        p_nu = polygamma(1, alpha + beta)
        h_aa = p_nu - polygamma(1, alpha)
        h_ab = p_nu
        h_bb = p_nu - polygamma(1, beta)
        if self._is_symmetric:
            return (h_aa + 2 * h_ab + h_bb,)
        return ((h_aa, h_ab), (h_ab, h_bb))


###############################################################################

# pylint: disable=unbalanced-tuple-unpacking

if __name__ == "__main__":

    def test1():
        """
        Test default parameters
        """
        btd = BetaDistribution()
        assert btd.num_parameters() == 2
        assert btd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
        _mu = btd.mean()
        assert _mu == DEFAULT_ALPHA / (DEFAULT_ALPHA + DEFAULT_BETA)
        assert btd.variance() == _mu * (1 - _mu) / (1 + DEFAULT_ALPHA + DEFAULT_BETA)
        print("Passed default parameter tests!")

    def test2():
        """
        Test fitting two observations
        """
        obs = np.array([0.25, 0.75])
        btd = BetaDistribution()
        res = btd.fit(obs)
        assert res["converged"]
        _mu = btd.mean()
        assert np.abs(_mu - np.mean(obs)) < 1e-6
        _alpha, _beta = btd.get_parameters()
        assert np.abs(_alpha - _beta) < 1e-6
        print("Passed simple fitting test!")

    def test3():
        """
        Test regression on two groups - means are complementary
        """
        # NOTE: Regression on obs=[0.25, 0.75] with covars=[-1, 1] converges to incorrect means!
        # Regression with covars=[(1,-1), (1,1)] diverges!
        obs_m1 = [0.2, 0.3]  # mean 0.25
        obs_p1 = [0.7, 0.8]  # mean 0.75
        obs = obs_m1 + obs_p1
        # NOTE: Regression with covars = [-1, -1, 1, 1] converges to incorrect means!
        covars_m1 = [(1, -1)] * len(obs_m1)
        covars_p1 = [(1, 1)] * len(obs_p1)
        covars = covars_m1 + covars_p1
        btr = BetaDistribution().regressor()
        res = btr.fit(obs, covars)
        assert res["converged"]
        mu_m1 = btr.mean(covars_m1[0])
        assert np.abs(mu_m1 - np.mean(obs_m1)) < 1e-4
        mu_p1 = btr.mean(covars_p1[0])
        assert np.abs(mu_p1 - np.mean(obs_p1)) < 1e-4
        print("Passed two-group regression")

    def test4():
        """
        Test regression on two groups - means are NOT complementary
        """
        obs_m1 = [0.1, 0.2, 0.3]  # mean 0.2
        obs_p1 = [0.6, 0.7, 0.8]  # mean 0.7
        covars_m1 = [(1, -1)] * len(obs_m1)
        covars_p1 = [(1, 1)] * len(obs_p1)
        obs = obs_m1 + obs_p1
        covars = covars_m1 + covars_p1
        btr = BetaDistribution().regressor()
        res = btr.fit(obs, covars)
        assert res["converged"]
        mu_m1 = btr.mean(covars_m1[0])
        assert np.abs(mu_m1 - np.mean(obs_m1)) < 1e-3
        mu_p1 = btr.mean(covars_p1[0])
        assert np.abs(mu_p1 - np.mean(obs_p1)) < 1e-3
        print("Passed two-group regression tests!")

    def test5():
        """
        Test regression with only bias
        """
        obs = [0.1, 0.2, 0.3] + [0.6, 0.7, 0.8]  # mean 0.2 vs mean 0.7
        covars0 = [1] * len(obs)
        btr = BetaDistribution().regressor()
        res = btr.fit(obs, covars0)
        phi1, phi2 = btr.get_parameters()
        weight1 = phi1[0]
        weight2 = phi2[0]
        btd = BetaDistribution()
        res = btd.fit(obs)
        assert res["converged"]
        alpha0, betaa0 = btd.get_parameters()
        alpha1, beta1 = btr.link_model().invert_transform(weight1, weight2)
        assert np.abs(alpha1 - alpha0) < 1e-12
        assert np.abs(beta1 - betaa0) < 1e-12
        eta1, eta2 = btr.link_model().apply_transform(alpha1, beta1)
        assert np.abs(eta1 - weight1) < 1e-15
        assert np.abs(eta2 - weight2) < 1e-15
        print("Passed bias-only regression tests!")

    def test6():
        """
        Test default symmetric parameter
        """
        btd = BetaDistribution(symmetric=True)
        assert btd.num_parameters() == 1
        assert btd.get_parameters() == (DEFAULT_ALPHA,)
        _mu = btd.mean()
        assert _mu == 0.5
        assert btd.variance() == _mu * (1 - _mu) / (1 + 2 * DEFAULT_ALPHA)
        print("Passed symmetric parameter tests!")

    # ================================================
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
