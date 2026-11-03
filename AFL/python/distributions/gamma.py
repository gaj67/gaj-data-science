"""
This module implements the gamma distribution.

The distributional parameters are alpha (shape) and beta (rate).

The link parameter is taken to be eta = log(alpha/beta) = log(mu),
which is not a natural parameter. The independent parameter is chosen
as psi = beta.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, polygamma

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    import imptools

    imptools.enable_relative()


from .core.data_types import (
    Value,
    Values,
    Values2d,
    Vector,
    mean_value,
)

from .core.distribution import StandardDistribution, set_link_model
from .core.link_models import LogRatioLink21b
from .core.optimiser import Data, Controls


#################################################################
# Gamma distribution


# Assume uniform prior:
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


@set_link_model(LogRatioLink21b)
# pylint: disable=too-many-ancestors
class GammaDistribution(StandardDistribution):
    """
    Implements the gamma probability distribution for a non-negative
    response variate, obs.
    """

    def __init__(self, alpha: Value = DEFAULT_ALPHA, beta: Value = DEFAULT_BETA):
        """
        Initialises the gamma distribution(s).

        Input:
            - alpha (scalar or vector): The shape parameter value(s).
            - beta (scalar or vector): The rate parameter value(s).
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
        return alpha / beta

    def variance(self) -> Value:
        alpha, beta = self.get_parameters()
        return alpha / beta**2

    # -----------------------------
    # GradientOptimisable interface

    def estimate_parameters(self, data: Data, controls: Controls) -> Values:
        # Nonlinear approximation
        mean = mean_value(data.weights, data.variate)
        m_ln = mean_value(data.weights, np.log(data.variate))
        ess = np.log(mean) - m_ln
        alpha = ((3 - ess) + np.sqrt((3 - ess) ** 2 + 24 * ess)) / (12 * ess)
        beta = alpha / mean
        if self.check_parameters(alpha, beta):
            return (alpha, beta)
        # Method of moments
        msq = mean_value(data.weights, data.variate**2)
        var = msq - mean**2
        alpha = mean**2 / var
        beta = mean / var
        if self.check_parameters(alpha, beta):
            return (alpha, beta)
        # Mean approximation
        alpha = DEFAULT_ALPHA
        beta = alpha / mean
        return (alpha, beta)

    def compute_scores(self, variate: Vector) -> Vector:
        alpha, beta = self.get_parameters()
        return gamma_dist.logpdf(variate, alpha, scale=1.0 / beta)

    def compute_gradients(self, variate: Vector) -> Values:
        # L = (
        #  alpha*ln(beta) - ln Gamma(alpha)
        #  + (alpha - 1)*ln(obs) - beta*obs
        # )
        alpha, beta = self.get_parameters()
        # d L / d alpha = ln obs - E[ln obs]
        g_a = np.log(beta) - digamma(alpha) + np.log(variate)
        # d L / d beta  = E[obs] - obs
        g_b = alpha / beta - variate
        return (g_a, g_b)

    def compute_hessian(self, variate: Vector) -> Values2d:
        alpha, beta = self.get_parameters()
        # d^2 L / d alpha^2 = -Var[ln obs]
        h_aa = -polygamma(1, alpha)  # pylint: disable=invalid-unary-operand-type
        # d^2 L / d beta^2 = -Var[-obs] = -Var[obs]
        h_bb = -alpha / beta**2
        # d^2 L / d alpha d beta = -Cov[ln obs, -obs] = Cov[ln obs, obs]
        h_ab = 1 / beta
        return ((h_aa, h_ab), (h_ab, h_bb))


###############################################################################


if __name__ == "__main__":

    def test1():
        """
        Test default parameter
        """
        gmd = GammaDistribution()
        assert gmd.get_parameters() == (DEFAULT_ALPHA, DEFAULT_BETA)
        assert gmd.mean() == DEFAULT_ALPHA / DEFAULT_BETA
        assert gmd.variance() == DEFAULT_ALPHA / DEFAULT_BETA**2
        print("Passed default parameter tests!")

    def test2():
        """
        Test fitting two observations
        """
        obs = np.array([1.0, 10])
        gmd = GammaDistribution()
        res = gmd.fit(obs)
        assert res["converged"]
        assert np.abs(gmd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting simple observations test!")

    def test3():
        """
        Test fitting multiple observations - possible divergence
        """
        obs = [3.4, 9.4, 9.6, 8.8, 8.4, 0.2, 4.3]
        gmd = GammaDistribution()
        res = gmd.fit(obs)
        assert res["converged"]
        assert np.abs(gmd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting problematic observations test!")

    def test4():
        """
        Test another problematic case
        """
        obs = [4.3, 3.8]
        gmd = GammaDistribution()
        res = gmd.fit(obs)
        assert res["converged"]
        assert np.abs(gmd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting second problematic observations test!")

    def test5():
        """
        Test fitting multiple observations
        """
        for num_obs in range(2, 11):
            obs = np.random.randint(1, 100, num_obs) * 0.1
            gmd = GammaDistribution()
            res = gmd.fit(obs)
            assert res["converged"]
            assert np.abs(gmd.mean() - np.mean(obs)) < 1e-6
        print("Passed fitting multiple observations tests!")

    def test6():
        """
        Test two-group regression
        """
        obs_m1 = [3.4, 0.2, 4.3]
        obs_p1 = [9.4, 9.6, 8.8, 8.4]
        obs = np.hstack((obs_m1, obs_p1))
        covars_m1 = [(1, -1)] * len(obs_m1)
        covars_p1 = [(1, 1)] * len(obs_p1)
        covars = np.vstack((covars_m1, covars_p1))
        gmr = GammaDistribution().regressor()
        res = gmr.fit(obs, covars)
        assert res["converged"]
        print("Passed two-group regression tests!")

    def test7():
        """
        Test regression with only bias
        """
        obs = np.hstack(([3.4, 0.2, 4.3], [9.4, 9.6, 8.8, 8.4]))
        covars0 = [1] * len(obs)
        gmr = GammaDistribution().regressor()
        res = gmr.fit(obs, covars0)
        assert res["converged"]
        wvec, beta = gmr.get_parameters()
        weight = wvec[0]
        gmd = GammaDistribution()
        res = gmd.fit(obs)
        assert res["converged"]
        (
            alpha0,
            beta0,
        ) = gmd.get_parameters()  # pylint: disable=unbalanced-tuple-unpacking
        weight0 = np.log(alpha0 / beta0)
        assert np.abs(weight - weight0) < 1e-15
        assert np.abs(beta - beta0) < 1e-15
        alpha1, beta1 = gmr.link_model().invert_transform(weight, beta)
        assert np.abs(alpha1 - alpha0) < 1e-15
        assert np.abs(beta1 - beta0) < 1e-15
        weight2, beta2 = gmr.link_model().apply_transform(alpha1, beta1)
        assert np.abs(weight - weight2) < 1e-15
        assert np.abs(beta - beta2) < 1e-15
        print("Passed bias-only regression tests!")

    ###############################################################

    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
