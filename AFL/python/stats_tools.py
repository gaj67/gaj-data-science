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


def fit_linear(x, y, weights=None):
    if weights is not None:
        w = np.asarray(weights)
    else:
        w = np.ones(len(x))
    beta = weighted_cov(w, x, y) / weighted_cov(w, x, x)
    alpha = weighted_mean(w, y) - beta * weighted_mean(w, x)
    return alpha, beta
