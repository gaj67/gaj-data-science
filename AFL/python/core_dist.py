"""
This module defines a base class for probability distribution functions (PDFs)
of a discrete or continuous scalar variate.

We assume for convenience that every PDF p(X|theta) is a member of "the" exponential family, 
and thus has natural parameters (eta) and natural variates (Y) that might not be the same 
as the default parameters (theta) and the given variate (X).

We also assume the implicit existence of an invertible link function (g) that maps the mean 
mu of the distribution into a so-called link parameter (also eta). Tne derivative of the 
log-likehood function with respect to this link parameter defines the difference between 
the so-called link variate (Y_eta) and its mean (mu_eta). Note that the link parameter is
not necessarily related to the natural distributional parameters, despite traditionally 
having the same label (i.e. both are called eta).
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Distribution(ABC):
    """
    A probability distribution of a scalar variate, X.
    """

    def __init__(self, *theta):
        """
        Initialises the distribution.
        
        Input:
            - theta (*float): The parameter values.
        """
        self._params = np.array(params)

    def parameters(self) -> ndarray:
        """
        Provides the values of the distributional parameters.
        
        Returns:
            - theta (array): An array of parameter values.
        """
        return self._params

    @abstractmethod
    def mean(self) -> float:
        """
        Computes the mean of the distribution.
        
        Returns:
            - mu (float): The mean value.
        """
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> float:
        """
        Computes the variance of the distribution.
        
        Returns:
            - sigma_sq (float): The variance.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_parameters(self) -> ndarray:
        """
        Computes the values of the natural parameters.
        
        Returns:
            - eta (array): An array of natural parameter values.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_variates(self, X: float) -> ndarray:
        """
        Computes the values of the natural variates from the given variate.
        
        Input:
            - X (float): The value of the distributional variate.
        Returns:
            - Y (array): An array of natural variate values.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_means(self) -> ndarray:
        """
        Computes the means of the natural variates.
        
        Returns:
            - mu (array): An array of natural variate means.
        """
        raise NotImplementedError

    @abstractmethod
    def natural_variances(self) -> ndarray:
        """
        Computes the variances and covariances of the natural variates.
        
        Returns:
            - Sigma (array): The variance matrix of the natural variates.
        """
        raise NotImplementedError

    @abstractmethod
    def link_parameter(self) -> float:
        """
        Computes the value of the link parameter from the mean via the link function. 
        
        Returns:
            - eta (float): The link parameter value.
        """
        raise NotImplementedError

    @abstractmethod
    def link_variate(self, X: float) -> float:
        """
        Computes the value of the link variate from the given variate.
        
        Input:
            - X (float): The value of the distributional variate.
        Returns:
            - Y (float): The link variate value.
        """
        raise NotImplementedError

    @abstractmethod
    def link_mean(self) -> float:
        """
        Computes the mean of the link variate.
        
        Returns:
            - mu (float): The link variate mean.
        """
        raise NotImplementedError

    @abstractmethod
    def link_variance(self) -> float:
        """
        Computes the variance of the link variate.
        
        Returns:
            - sigma_sq (float): The link variate variance.
        """
        raise NotImplementedError
