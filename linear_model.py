import numpy as np

from base import BaseEstimator

class LinearRegression(BaseEstimator):
    """
    Estimator that can compute linear regression
    """
    def __init__(self, fit_intercept=True, normalise=False, copy_X=True, n_jobs=None):
        
        # intialise parameters
        self.fit_intercept=fit_intercept
        self.normalise=normalise
        self.copy_X=copy_X
        self.n_jobs=n_jobs
        
        # initialise attributes
        self.coef_ = np.empty((1))
        self.rank_ = 0
        self.singular_ = np.empty((1))
        self.intercept_ = np.empty((1))
    
    def fit(self, X, y, sample_weight=None):
        """
        fit method that fits the parameters coef_ and intercept_ to the training set X
        """
        # no implementation of sample_weight attribute
        return self

    def get_params(self, deep=True):
        """
        get parameters for this estimator
        """
        # no implementation of deep attribute
        return dict([["coef", self.coef_],
                     ["intercept", self.intercept_],
                     ["rank", self.rank_],
                     ["singular", self.singular_],
                     ])

    def predict(self, X):
        """
        predict using the linear model
        """
        pass

    def score(self, X, y, sample_weight=None):
        """
        return the coefficient of determination R^2 of the prediction
        """
        pass

    def set_params(self, **params):
        """
        set the parameters of this estimator
        """