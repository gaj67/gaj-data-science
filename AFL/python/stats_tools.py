"""
This module provides some simple tools for statistical analysis and model fitting.
"""

import numpy as np


def logistic(x):
    """
    Computes the logistic transformation for each input.

    Input:
        - x (float or array-like): The value(s) to be  transformed.
    Returns:
        - p (float or ndarray): The transformed value(s).
    """
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    """
    Computes the logit transformation for each input.

    Input:
        - p (float or array-like): The value(s) to be  transformed.
    Returns:
        - x (float or ndarray): The transformed value(s).
    """
    p = np.asarray(p)
    return np.log(p / (1.0 - p))


def weighted_mean(weights, x, y=None):
    w = np.asarray(weights)
    if y is None:
        return np.mean(w * x) / np.mean(w)
    else:
        return np.mean(w * x * y) / np.mean(w)


def weighted_cov(weights, x, y):
    w = np.asarray(weights)
    m_xy = weighted_mean(w, x, y)
    m_x = weighted_mean(w, x)
    m_y = weighted_mean(w, y)
    return m_xy - m_x * m_y


def weighted_var(weights, x):
    w = np.asarray(weights)
    m_x2 = weighted_mean(w, x, x)
    m_x = weighted_mean(w, x)
    return m_x2 - m_x**2


def fit_linear(x, y, weights=None):
    if weights is not None:
        w = np.asarray(weights)
    else:
        w = np.ones(len(x))
    beta = weighted_cov(w, x, y) / weighted_cov(w, x, x)
    alpha = weighted_mean(w, y) - beta * weighted_mean(w, x)
    return alpha, beta


def partition_points(data, num_bins=20, axis=0, adj=1e-3):
    """
    Partitions a collection of points into discrete bins.

    Inputs:
        - data (array): An N x V array of N points in V dimensions.
        - num_bins (int): The number of partitions.
        - axis (int): The data dimension on which to partition.
        - adj (float): The size of the adjustment to the minimum and maximum values.

    Returns:
        - bins (list of array): The partitioned data.
    """
    X = data[:, axis]
    bin_edges = np.linspace(min(X) - adj, max(X) + adj, num_bins)
    indexes = np.digitize(X, bin_edges) - 1
    bins = [None] * num_bins
    for i in range(num_bins):
        ind = indexes == i
        bins[i] = data[ind, :]
    return bins


def aggregate_bins(bins, num_data=10):
    """
    Aggregates bins until either all bins contain at least the minimum number
    of data points, or else there is a single bin remaining.

    Inputs:
        - bins (list of array): The unaggregated bins.
        - num_data (int): The minimum bin size.
    Returns:
        - bins (list of array): The aggregated bins.
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
