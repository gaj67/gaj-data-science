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


def summarise_bins(bins):
    """
    Computes the weighted, two-dimensional mean and variance of each bin.
    The bins are assumed to have already been aggregated to contain sufficient points
    for which to compute the bin statistics.

    Inputs:
        - bins (list of array): The aggregated bins in format [[X_data, Y_data, weights]].
    Returns:
        - n_points (array of int): The number of points in each bin.
        - x_means (array of float): The mean of the x-coordinate in each bin.
        - y_means (array of float): The mean of the y-coordinate in each bin.
        - x_vars (array of float): The variance of the x-coordinate in each bin.
        - y_vars (array of float): The variance of the y-coordinate in each bin.
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
        X = bin_data[:,0]
        Y = bin_data[:,1]
        weights = bin_data[:,2]
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


def summarise_data(X, Y, weights=None):
    """
    Performs standardised binning of the data along the x-coordinate,
    and computes the two-dimensional summary statistics of each bin.

    Inputs:
        - X (array): The array of x-coordinates.
        - Y (array): The array of y-coordinates.
        - weights (array): An optional array specifying the weight of each point.
            If this is not provided, then every point is weighted equally.
    Returns:
        - n_points (array of int): The number of points in each bin.
        - x_means (array of float): The mean of the x-coordinate in each bin.
        - y_means (array of float): The mean of the y-coordinate in each bin.
        - x_se (array of float): The standard error of the x-coordinate in each bin.
        - y_se (array of float): The standard error of the y-coordinate in each bin.
    """
    if weights is None:
        weights = np.ones(len(X))
    data = np.column_stack([X, Y, weights])
    bins = partition_points(data)
    bins = aggregate_bins(bins)
    n_points, x_means, y_means, x_vars, y_vars = summarise_bins(bins)
    x_se = np.sqrt(x_vars / n_points)
    y_se = np.sqrt(y_vars / n_points)
    return n_points, x_means, y_means, x_se, y_se
