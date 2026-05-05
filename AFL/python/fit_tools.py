"""
This module provides some simple tools for statistical analysis and model fitting.
"""

from typing import Optional
from numpy.typing import ArrayLike
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import stats_tools


def LQC_regression(
    X: ArrayLike, Y: ArrayLike, weights: Optional[ArrayLike] = None,
    linear: bool = False,
    quadratic: bool = False,
    cubic: bool = False,
    intercept: bool = True
) -> list[tuple]:
    """
    Optionally fits linear, quadratic and/or cubic regression,
    in a form reaady for plotting. By default, no regression is assumed.
    
    Inputs:
      - X (list or array): The predictor values.
      - Y (list or array): The response values.
      - weights (list or array, optional): The weights for each data point.
      - linear (bool, optional): Indicates whether or not to include linear regression.
      - quadratic (bool, optional): Indicates whether or not to include quadratic regression.
      - cubic (bool, optional): Indicates whether or not to include cubic regression.
      - intercept (bool, optional): Indicates whether or not to include an intercept term.
          Defaults to True.
    Returns:
      - points (list of tuple): A list of XY pairs of plotting points, containing the
          original data points plus any regressionn fits. Each tuple also specifies
          an optional legend label and an optional colour/point symbol.
    """
    # Summarise data
    X = np.asarray(X)
    Y = np.asarray(Y)
    points = [(X, Y, 'data', '.')]
    # Setup regression
    if weights is None:
        weights = np.ones(len(X))
    X0 = np.linspace(min(X), max(X), 100)
    # Fit linear regression
    if linear:
        lr = LinearRegression(fit_intercept=intercept).fit(
            stats_tools.no_intercept(X),
            Y, weights
        )
        Y1 = lr.predict(stats_tools.no_intercept(X0))
        points.append((X0, Y1, 'linear', 'k'))
    # Fit quadratic regression
    if quadratic:
        lr = LinearRegression(fit_intercept=intercept).fit(
            stats_tools.no_intercept(X, X**2),
            Y, weights
        )
        Y2 = lr.predict(stats_tools.no_intercept(X0, X0**2))
        points.append((X0, Y2, 'quadratic', 'purple'))
    # Fit cubic regression
    if cubic:
        lr = LinearRegression(fit_intercept=intercept).fit(
            stats_tools.no_intercept(X, X**2, X**3),
            Y, weights
        )
        Y3 = lr.predict(stats_tools.no_intercept(X0, X0**2, X0**3))
        points.append((X0, Y3, 'cubic', 'g'))
    # Return fits
    return points


def LQC_mean_regression(
    X: ArrayLike, Y: ArrayLike,
    linear: bool = False,
    quadratic: bool = False,
    cubic: bool = False,
    intercept: bool = True
) -> list[tuple]:
    """
    Optionally fits linear, quadratic and/or cubic regression
    to the XY-centroids of X-partitions of the data, in a form
    reaady for plotting. By default, no regression is assumed.
    
    Inputs:
      - X (list or array): The predictor values.
      - Y (list or array): The response values.
      - linear (bool, optional): Indicates whether or not to include linear regression.
      - quadratic (bool, optional): Indicates whether or not to include quadratic regression.
      - cubic (bool, optional): Indicates whether or not to include cubic regression.
      - intercept (bool, optional): Indicates whether or not to include an intercept term.
          Defaults to True.
    Returns:
      - points (list of tuple): A list of XY pairs of plotting points, containing the
          centroid data points, plus any regressionn fits, followed by +/- two
          standard errors in Y. Each tuple also specifies
          an optional legend label and an optional colour/point symbol.
    """
    # Summarise data
    stats = stats_tools.summarise_data(X, Y)
    n_points, x_means, y_means, x_se, y_se = stats
    # Fit models
    points = LQC_regression(
        x_means, y_means, n_points,
        linear, quadratic, cubic,
        intercept
    )
    # Add raw data
    _ = points.pop(0)
    points.insert(0, (x_means, y_means, 'mean', '*'))
    points.insert(0, (X, Y, 'data', '.'))
    # Add standard errors
    points.append((x_means, y_means + 2 * y_se, '2 s.e.',  'lightgrey'))
    points.append((x_means, y_means - 2 * y_se, None,  'lightgrey'))
    # Return points
    return points


def distributional_regression(X: ArrayLike, Y: ArrayLike, *dists) -> list[tuple]:
    """
    Optionally fits distributional regression to the data, in a form
    reaady for plotting. By default, no regression is assumed.
    
    Inputs:
      - X (list or array): The predictor values.
      - Y (list or array): The response values.
      - dists (tuple of Distribution): A collection of Distribution objects,
          ready for fitting to the data. As a special variant, the distribution
          may include a plotting symbol in the format (Distribution, symbol).
    Returns:
      - points (list of tuple): A list of XY pairs of plotting points, containing the
          original data points, plus any distributional regression fits.
          Each tuple also specifies an optional legend label and an optional plotting symbol.
    """
    # Fit distributional regression models
    points = [(X, Y, 'data', '.')]
    X0 = np.linspace(min(X), max(X), 100)
    for dist in dists:
        if isinstance(dist, tuple):
            dist, symb = dist
        else:
            symb = None
        dr = dist.regressor()
        res = dr.fit(
            Y, stats_tools.add_intercept(X)
        )
        Yd = dr.mean(stats_tools.add_intercept(X0))
        dn = dist.__class__.__name__
        if dn.endswith("Distribution"):
            dn = dn[0:-12]
        points.append((X0, Yd, dn, symb))
    # Return points
    return points


def plot_regression_points(
    points: list[tuple],
    xlabel:  str = "", ylabel: str = "",
    title: str = "",
    xticks: Optional[ArrayLike] = None,
    legend_loc: str = "best"
):
    """
    Inputs:
      - points (list of tuple): A list of XY pairs of plotting points, 
          each including an optional legend label and an optional plotting symbol.
      - xlabel (str, optional): The X-axis label 
      - ylabel (str, optional): The Y-axis label 
      - title (str, optional):  The figure title.
      - xticks (list or array, optional)): The X-tick points.
      - legend_loc (str, optional): The position off the legend.
    """
    labels = []
    for x, y, label, symb in points:
        if label is not None:
            labels.append(label)
        if symb is None:
            plt.plot(x, y)
        else:
            plt.plot(x, y, symb)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xticks is not None:
        plt.xticks(xticks)
    plt.legend(labels, loc=legend_loc)
    plt.show()
