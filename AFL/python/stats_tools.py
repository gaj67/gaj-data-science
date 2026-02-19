"""
This module provides some simple tools for statistical analysis and model fitting.
"""

from typing import Optional, List, Tuple
from numpy import ndarray
from numpy.typing import ArrayLike
import numpy as np
import scipy.stats as sci_stats
import warnings


def logistic(x: ArrayLike) -> ArrayLike:
    """
    Computes the logistic transformation for each input.

    Input:
        - x (float or array-like): The value(s) to be  transformed.
    Returns:
        - p (float or ndarray): The transformed value(s).
    """
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: ArrayLike) -> ArrayLike:
    """
    Computes the logit transformation for each input.

    Input:
        - p (float or array-like): The value(s) to be  transformed.
    Returns:
        - x (float or ndarray): The transformed value(s).
    """
    p = np.asarray(p)
    return np.log(p / (1.0 - p))


def weighted_mean(
    weights: ArrayLike, x: ArrayLike, y: Optional[ArrayLike] = None
) -> float:
    """
    Computes the weighted mean of the x-variate. If a y-variate is also supplied,
    then the weighted mean on the product of the variates will be computed instead.

    Input:
        - weights (array-like): The weights of the variate(s).
        - x (array-like): The x-variate.
        - y (array-like): The optional y-variate.
    Returns:
        - mean (float): The weighted mean.
    """
    w = np.asarray(weights)
    if y is None:
        return np.mean(w * x) / np.mean(w)
    else:
        return np.mean(w * x * y) / np.mean(w)


def weighted_cov(weights: ArrayLike, x: ArrayLike, y: ArrayLike) -> float:
    """
    Computes the weighted covariance of the x- and y-variates.

    Input:
        - weights (array-like): The weights of the variates.
        - x (array-like): The x-variate.
        - y (array-like): The y-variate.
    Returns:
        - cov (float): The weighted covariance.
    """
    w = np.asarray(weights)
    m_xy = weighted_mean(w, x, y)
    m_x = weighted_mean(w, x)
    m_y = weighted_mean(w, y)
    return m_xy - m_x * m_y


def weighted_var(weights: ArrayLike, x: ArrayLike) -> float:
    """
    Computes the weighted variance of the x-variate.

    Input:
        - weights (array-like): The weights of the variates.
        - x (array-like): The x-variate.
    Returns:
        - var (float): The weighted variance.
    """
    w = np.asarray(weights)
    m_x2 = weighted_mean(w, x, x)
    m_x = weighted_mean(w, x)
    v = max(0, m_x2 - m_x**2)
    return v


def fit_linear(x, y, weights=None):
    if weights is not None:
        w = np.asarray(weights)
    else:
        w = np.ones(len(x))
    beta = weighted_cov(w, x, y) / weighted_cov(w, x, x)
    alpha = weighted_mean(w, y) - beta * weighted_mean(w, x)
    return alpha, beta


def add_intercept(column, *columns) -> ndarray:
    """
    Wraps the given vector(s) into a column matrix suitable for covariates
    in a regression model. The first column is an additional unit vector
    representing an intercept or bias term.

    Inputs:
        - column (array-like): The values of the first covariate.
        - columns (tuple of array-like): The optional values of subsequent
            covariates.
    Returns:
        - matrix (ndarray): The two-dimensional covariate matrix.
    """
    const = np.ones(len(column))
    return np.stack((const, column) + columns, axis=1)


def no_intercept(column, *columns) -> ndarray:
    """
    Wraps the given vector(s) into a column matrix suitable for covariates
    in a regression model. No intercept or bias term is added.

    Inputs:
        - column (array-like): The values of the first covariate.
        - columns (tuple of array-like): The optional values of subsequent
            covariates.
    Returns:
        - matrix (ndarray): The two-dimensional covariate matrix.
    """
    return np.stack((column,) + columns, axis=1)


def partition_points(
    data: ndarray, num_bins: int = 20, axis: int = 0, adj: float = 1e-3
) -> List[ndarray]:
    """
    Partitions a collection of points into discrete, regularly spaced bins.
    If no points fall within a given bin, then the data for that bin will have zero size.

    Inputs:
        - data (ndarray): An N x V array of N points in V dimensions.
        - num_bins (int): The number of partitions.
        - axis (int): The data dimension on which to partition.
        - adj (float): The size of the adjustment to the minimum and maximum values.

    Returns:
        - bins (list of ndarray): The partitioned data.
    """
    X = data[:, axis]
    bin_edges = np.linspace(min(X) - adj, max(X) + adj, num_bins)
    indexes = np.digitize(X, bin_edges) - 1
    bins = [None] * num_bins
    for i in range(num_bins):
        ind = indexes == i
        bins[i] = data[ind, :]
    return bins


def aggregate_bins(bins: List[ndarray], num_data: int = 10) -> List[ndarray]:
    """
    Aggregates small-sized bins with adjacent bins until either all bins
    contain at least the minimum number of data points, or else there is
    only a single bin remaining.

    Inputs:
        - bins (list of ndarray): The unaggregated bins.
        - num_data (int): The minimum bin size.
    Returns:
        - bins (list of ndarray): The aggregated bins.
    """

    def _merge(i, j):
        # XXX i adjacent to j, i.e. abs(i - j) == 1
        return (
            bins[: min(i, j)]
            + [np.concatenate([bins[i], bins[j]])]
            + bins[max(i, j) + 1 :]
        )

    n_bins = len(bins)
    i = 0
    while n_bins >= 2 and i < n_bins:
        b = bins[i]
        if len(b) >= num_data:
            i += 1
            continue
        if i == 0:
            # Extreme left - merge right
            bins = _merge(0, 1)
        elif i == n_bins - 1:
            # Extreme right - merge left
            bins = _merge(i, i - 1)
        elif len(bins[i - 1]) <= len(bins[i + 1]):
            # Merge left to smaller bin
            bins = _merge(i, i - 1)
        else:
            # Merge right to smaller bin
            bins = _merge(i, i + 1)
        n_bins -= 1
    return bins


def summarise_bins(bins: List[ndarray]) -> Tuple:
    """
    Computes the weighted, two-dimensional mean and variance of each bin.
    The bins are assumed to have already been aggregated to contain sufficient points
    for which to compute the bin statistics.

    Inputs:
        - bins (list of ndarray): The aggregated bins in format [[X_data, Y_data, weights]].
    Returns:
        - n_points (ndarray of int): The number of points in each bin.
        - x_means (ndarray of float): The mean of the x-coordinate in each bin.
        - y_means (ndarray of float): The mean of the y-coordinate in each bin.
        - x_vars (ndarray of float): The variance of the x-coordinate in each bin.
        - y_vars (ndarray of float): The variance of the y-coordinate in each bin.
    """
    n_points = []
    x_means = []
    y_means = []
    x_vars = []
    y_vars = []

    for i in range(len(bins)):
        bin_data = bins[i]
        if len(bin_data) == 0:
            continue
        X = bin_data[:, 0]
        Y = bin_data[:, 1]
        weights = bin_data[:, 2]
        n_points.append(len(X))
        x_means.append(weighted_mean(weights, X))
        y_means.append(weighted_mean(weights, Y))
        x_vars.append(weighted_var(weights, X))
        y_vars.append(weighted_var(weights, Y))

    n_points = np.array(n_points)
    x_means = np.array(x_means)
    y_means = np.array(y_means)
    x_vars = np.array(x_vars)
    y_vars = np.array(y_vars)
    return n_points, x_means, y_means, x_vars, y_vars


def summarise_data(
    X: ArrayLike, Y: ArrayLike = None,
    weights: Optional[ArrayLike] = None, 
    num_bins: int = 20, return_vars: bool = False
) -> Tuple:
    """
    Performs standardised binning of the data with respect to the x-variate,
    and computes the two-dimensional summary statistics of each bin.

    Inputs:
        - X (array-like): The x-variate.
        - Y (array-like): The optional y-variate.
        - weights (array-like): The optional weights of the data.
            If these are not provided, then every point is weighted equally.
        - num_bins (int): The number of partitions.
        - return_vars (bool): An optional flag indicating whether to return
            variances (if True) or standard errors (if False).
            Defaults to False.
    Returns:
        - n_points (ndarray of int): The number of points in each bin.
        - x_means (ndarray of float): The mean of the x-variate in each bin.
        - y_means (ndarray of float): The mean of the y-variate in each bin.
        - x_se | x_vars (ndarray of float): The standard error or variance of the x-variate in each bin.
        - y_se | y_vars (ndarray of float): The standard error or variance of the y-variate in each bin.
    """
    if weights is None:
        weights = np.ones(len(X))
    if Y is None:
        Y = np.zeros(len(X))
    data = np.column_stack([X, Y, weights])
    bins = partition_points(data, num_bins = num_bins)
    bins = aggregate_bins(bins)
    n_points, x_means, y_means, x_vars, y_vars = summarise_bins(bins)
    if return_vars:
        return n_points, x_means, y_means, x_vars, y_vars
    x_se = np.sqrt(x_vars / n_points)
    y_se = np.sqrt(y_vars / n_points)
    return n_points, x_means, y_means, x_se, y_se


def R_squared(
    Y_obs: ArrayLike, Y_pred: ArrayLike, weights: Optional[ArrayLike] = None
) -> float:
    """
    Computes the weighted R^2 value, i.e. the proportion of total, weighted
    variance that is explained by the predictive model.
    If no weights are provided, then all data are weighted equally.

    Inputs:
        - Y_obs (array-like): The observed values.
        - Y_pred (array-like): The predicted values.
        - weights (array-like): The optional weights of the data.
    Returns:
        - r2 (float): The R^2 value.
    """
    if weights is None:
        W = np.ones(len(Y_obs))
    else:
        W = np.asarray(weights)
    Y = np.asarray(Y_obs)
    Y_bar = weighted_mean(W, Y)
    v_baseline = weighted_mean(W, (Y - Y_bar) ** 2)
    v_model = weighted_mean(W, (Y - Y_pred) ** 2)
    r2 = 1 - v_model / v_baseline
    return r2


def cross_entropy(
    Y_obs: ArrayLike, Y_pred: ArrayLike, weights: Optional[ArrayLike] = None
) -> float:
    """
    Computes the weighted-average cross-entropy score.
    If no weights are provided, then all data are weighted equally.

    Inputs:
        - Y_obs (array-like): The observed proportions.
        - Y_pred (array-like): The predicted proportions.
        - weights (array-like): The optional weights of the data.
    Returns:
        - xent (float): The cross-entropy score.
    """
    Y = np.asarray(Y_obs)
    Yh = np.asarray(Y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = Y * np.log(Yh) + (1 - Y) * np.log(1 - Yh)
    if weights is None:
        return -np.mean(X)
    else:
        return -np.mean(X * weights) / np.mean(weights)


def binary_accuracy(
    Y_obs: ArrayLike, Y_pred: ArrayLike, weights: Optional[ArrayLike] = None
) -> float:
    """
    Computes the weighted-average accuracy of binary predictions.
    If no weights are provided, then all data are weighted equally.

    Inputs:
        - Y_obs (array-like): The observed indicators or proportions of "successes".
        - Y_pred (array-like): The predicted probabilities of "success".
        - weights (array-like): The optional weights of the data.
    Returns:
        - acc (float): The accuracy score.
    """
    Y = np.asarray(Y_obs)
    Yh = np.asarray(Y_pred)
    S = np.zeros(len(Y))
    # Measure observed "successes":
    ind = Y > 0.5
    S[ind] = Yh[ind] > 0.5
    # Measure observed "failures":
    ind = Y < 0.5
    S[ind] = Yh[ind] < 0.5
    # Allow for uncertain observations - all predictions will be half-right:
    S[Y == 0.5] = 0.5
    # Allow for uncertain predictions - random tie-breaking will be half-right:
    S[Yh == 0.5] = 0.5
    # Measure predicted accuracy
    if weights is None:
        return np.mean(S)
    else:
        return np.mean(S * weights) / np.mean(weights)


def uniform_bootstraps(data: ArrayLike, stats_fn, num_bootstraps: int = 100, proportion: float = 1) -> ndarray:
    """
    Creates bootstrap samples by uniform resampling of the data.
    For each bootstrap sample, the required statistics are computed.
    
    Inputs:
        - data (array-like): The collection of N data from which to resample.
        - stats_fn (function): A function to compute an S-tuple of statistics
            from a given bootstrap sample.
        - num_bootstraps (int): The number B of bootstrap samples.
        - proportion (int): The proportion p of data to resample for
            each bootstrap sample.
    Returns:
        - bootstraps (ndarray): The B x S array of bootstrap statistics.
    """
    data = np.asarray(data)
    N = len(data)
    M = int(proportion * N)
    bootstraps = [None] * num_bootstraps
    for i in range(num_bootstraps):
        idx = np.random.randint(N, size = M)
        bootstraps[i] = stats_fn(data[idx])
    return np.array(bootstraps)


def ranged_bootstraps(data: ArrayLike, stats_fn, num_bootstraps: int = 100, min_data: int = 10) -> ndarray:
    """
    Creates bootstrap samples by resampling from random ranges of the data.
    For each bootstrap sample, the required statistics are computed.
    
    Inputs:
        - data (array-like): The collection of N data from which to resample.
        - stats_fn (function): A function to compute an S-tuple of statistics
            from a given bootstrap sample.
        - num_bootstraps (int): The number B of bootstrap samples.
        - min_data (int): The minimum size of a viable bootstrap sample.
    Returns:
        - bootstraps (ndarray): The B x S array of bootstrap statistics.
    """
    L, U = min(data), max(data)
    bootstraps = [None] * num_bootstraps
    i = 0
    while i < num_bootstraps:
        p1 = np.random.randint(L, U)
        p2 = np.random.randint(L, U)
        if p1 == p2:
            # Reject the sample
            continue
        ind = (data >= min(p1, p2)) & (data <= max(p1, p2))
        if sum(ind) < min_data:
            # Reject the sample
            continue
        bootstraps[i] = stats_fn(data[ind])
        i += 1
    return np.array(bootstraps)


def negative_binomial_bootstraps(num_trials: int, num_events: int, num_bootstraps: int) -> ndarray:
    """
    Creates bootstrap samples of Bernoulli trials using negative binomial resampling.
    The proportion of special events in each bootstrap sample is then computed.
    
    Consider a notional collection of N = X + S Bernoulli trials containing S special events, 
    e.g. "successes" or "failures", such that the empirical probability of a special event
    is Y = S / N. Each bootstrap sample is obtained by repeated resampling with
    replacement, using the probability Y, until S special events have been attained.
    For the ith bootstrap sample, there will be a total of N_i = X_i + S resamples, giving
    an estimated proportion of Y_i = S / N_i.
    
    Inputs:
        - num_trials (int): The number of trials.
        - num_events (int): The number of special events.
        - num_bootstraps (int): The number of bootstrap samples.
    Returns:
        - proportions (ndarray of float): The proportion of special events in each
            bootstrap sample.
    """
    p_event = num_events / num_trials
    bootstrap_trials = sci_stats.nbinom.rvs(
        num_events, p_event, size=num_bootstraps
    )
    return num_events / (bootstrap_trials + num_events)


def negative_binomial_densities(num_trials: int, num_events: int, proportions: ndarray) -> ndarray:
    """
    Creates approximate probability density values of estimated proportions using a continuous
    transformation of the negative binomial distribution.
    
    If X + N is the total number of trials required to attain N special events (inclusive),
    then the estimated proportion of events is Y = N / (X + N), such that X = N * (1 - Y) / Y.
    The Jacobian of the transformation is |dY/dX| = N / (X + N)**2 = Y**2 / N.
    
    Inputs:
        - num_trials (int): The number of trials.
        - num_events (int): The number of special events.
        - proportions (ndarray of float): Estimates of the proportion of special events.
    Returns:
        - densities (ndarray of float): The approximate density of each estimate.
    """
    variates = np.round(num_events * (1 - proportions) / proportions)
    jacobian =  proportions**2 / num_events
    p_event = num_events / num_trials
    p = sci_stats.nbinom.pmf(variates, num_events, p_event) / jacobian
    return p
