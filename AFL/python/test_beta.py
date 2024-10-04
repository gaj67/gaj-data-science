import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import loggamma, digamma, polygamma

from core_dist import no_intercept, add_intercept


def log_prob(X, alpha, beta):
    v1 = beta_dist.logpdf(X, alpha, beta)
    print("DEBUG: v1 =", v1)
    v2 = (
        (alpha - 1) * np.log(X)
        + (beta - 1) * np.log(1 - X)
        + loggamma(alpha + beta)
        - loggamma(alpha)
        - loggamma(beta)
    )
    print("DEBUG: v2 =", v2)
    return v2


def convert_to_internal(alpha, beta):
    eta = np.log(alpha / beta)
    psi = alpha
    return eta, psi


def convert_from_internal(eta, psi):
    alpha = psi
    beta = alpha * np.exp(-eta)
    return alpha, beta


def get_gradient(X, alpha, beta):
    eta = np.log(alpha / beta)
    Y_alpha, Y_beta = np.log(X), np.log(1 - X)
    Y_eta = -beta * Y_beta
    Y_psi = Y_alpha + beta / alpha * Y_beta
    d_nu = digamma(alpha + beta)
    mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
    mu_beta = digamma(beta) - d_nu  # E[Y_beta]
    mu_eta = -beta * mu_beta  # E[Y_eta]
    mu_psi = mu_alpha + beta / alpha * mu_beta  # E[Y_psi]
    return np.array([Y_eta - mu_eta, Y_psi - mu_psi])


def get_neg_hessian(X, alpha, beta):
    # Compute variances
    cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
    v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
    v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
    v_eta = beta**2 * v_beta  # Var[Y_eta]
    k = beta / alpha
    v_psi = v_alpha + k**2 * v_beta + 2 * k * cov_ab  # Var[Y_psi]
    cov_pe = -beta * (cov_ab + k * v_beta)  # Cov[Y_psi, Y_eta]
    Sigma = np.array([[v_eta, cov_pe], [cov_pe, v_psi]])
    # Adjust for Hessian
    g_eta, g_psi = get_gradient(X, alpha, beta)
    Sigma[0, 1] -= g_eta / alpha
    Sigma[1, 0] = Sigma[0, 1]
    Sigma[0, 0] += g_eta
    return Sigma


def get_regression_gradient(X, Z, psi, phi):
    eta = Z @ phi
    alpha, beta = convert_from_internal(eta, psi)
    g_eta, g_psi = get_gradient(X, alpha, beta)
    g_phi = g_eta @ Z / len(g_eta)
    g_psi = np.mean(g_psi)
    return g_psi, g_phi  # scalar, vector


def get_regression_neg_hessian(X, Z, psi, phi):
    eta = Z @ phi
    alpha, beta = convert_from_internal(eta, psi)
    Sigma = get_neg_hessian(X, alpha, beta)
    XXX

def get_regression_score(X, Z, psi, phi):
    eta = Z @ phi
    alpha, beta = convert_from_internal(eta, psi)
    return np.mean(log_prob(X, alpha, beta))


    def invert_parameters(self, eta: Value, psi: Value) -> Values:
        alpha = psi
        beta = alpha * np.exp(-eta)
        return (alpha, beta)

    def internal_variates(self, X: Value) -> Values:
        alpha, beta = self.parameters()
        Y_alpha, Y_beta = np.log(X), np.log(1 - X)
        Y_eta = -beta * Y_beta
        Y_psi = Y_alpha + beta / alpha * Y_beta
        return (Y_eta, Y_psi)

    def internal_means(self) -> Values:
        alpha, beta = self.parameters()
        d_nu = digamma(alpha + beta)
        mu_alpha = digamma(alpha) - d_nu  # E[Y_alpha]
        mu_beta = digamma(beta) - d_nu  # E[Y_beta]
        mu_eta = -beta * mu_beta  # E[Y_eta]
        mu_psi = mu_alpha + beta / alpha * mu_beta  # E[Y_psi]
        return (mu_eta, mu_psi)

    def internal_variances(self) -> ndarray:
        alpha, beta = self.parameters()
        cov_ab = -polygamma(1, alpha + beta)  # Cov[Y_alpha, Y_beta]
        v_alpha = polygamma(1, alpha) + cov_ab  # Var[Y_alpha]
        v_beta = polygamma(1, beta) + cov_ab  # Var[Y_beta]
        v_eta = beta**2 * v_beta  # Var[Y_eta]
        k = beta / alpha
        v_psi = v_alpha + k**2 * v_beta + 2 * k * cov_ab  # Var[Y_psi]
        cov_pe = -beta * (cov_ab + k * v_beta)  # Cov[Y_psi, Y_eta]
        return np.array([[v_eta, cov_pe], [cov_pe, v_psi]])

    def initialise_parameters(self, X: Value, W: Value, **kwargs: dict):
        mu = guard_prob(weighted_mean(W, X))
        sigma_sq = weighted_var(W, X)
        if sigma_sq > 0.01:
            nu = mu * (1 - mu) / sigma_sq
            alpha = mu * nu
            beta = nu - alpha
        else:
            beta = DEFAULT_BETA
            alpha = beta * mu / (1 - mu)
        self.set_parameters(alpha, beta)


###############################################################################

# Simulate regression data
X = np.array([0.25, 0.75])
Z = add_intercept([-1, +1])

# Initialise distributional parameters
alpha = 1
mu = np.mean(X)
beta = alpha * (1 - mu) / mu

score0 = np.mean(log_prob(X, alpha, beta))

# Initialise regression parameters
phi = np.zeros(2)

# Compute initial conditions
nH = get_regression_lhs(X, Z, alpha, phi)
g = get_regression_rhs(X, Z, alpha, phi)
d_int = solve(nH, g)