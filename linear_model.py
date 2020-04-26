import numpy as np

from base import BaseEstimator

class LinearRegression(BaseEstimator):
    """
    Estimator that can compute linear regression
    """
    def __init__(self, copy_X=True, fit_intercept=True, n_jobs=None, normalise=False):
        # intialise parameters
        self.copy_X=copy_X
        self.fit_intercept=fit_intercept
        self.n_jobs=n_jobs
        self.normalise=normalise
        # initialise attributes
        self.coef_ = np.empty((1))
        self.rank_ = 0
        self.singular_ = np.empty((1))
        self.intercept_ = 0
    
    def __repr__(self):
        """
        copy of scikit-learn representation of a LinearRegression estimator
        """
        return f"LinearRegression(copy_X={self.copy_X}," + \
                                f"fit_intercept={self.fit_intercept}," + \
                                f"n_jobs={self.n_jobs}," + \
                                f"normalise={self.normalise})"
    
    def fit(self, X, y, sample_weight=None):
        """
        fit method that fits the parameters coef_ and intercept_ to the training set X
        """
        # no implementation of sample_weight attribute
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0
        return self

    def get_params(self, deep=True):
        """
        get parameters for this estimator
        """
        return dict([["copy_X", self.copy_X],
                     ["fit_intercept", self.fit_intercept],
                     ["n_jobs", self.n_jobs],
                     ["normalise", self.normalise],
                     ])

    def predict(self, X):
        """
        predict using the linear model
        """
        try:
            predictions = X@self.coef_ + self.intercept_
        except Exception as ex:
            print(ex)
            print("Error while predicting, thus predicting 0s instead...")
            predictions = np.zeros(X.shape[0])
        return predictions


    def score(self, X, y, sample_weight=None):
        """
        return the coefficient of determination R^2 of the prediction
        """
        pass

    def set_params(self, **params):
        """
        set the parameters of this estimator
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

# for testing purposes
if __name__=="__main__":
    from sklearn import linear_model 
    lin_reg_cb = LinearRegression()
    lin_reg_sk = linear_model.LinearRegression()