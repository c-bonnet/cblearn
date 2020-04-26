import numpy as np

from base import BaseEstimator

class LinearRegression(BaseEstimator):
    """
    Estimator that can compute linear regression
    """
    def __init__(self, normalise=False):
        # intialise parameters
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
        return f"LinearRegression(normalise={self.normalise})"
    
    def fit(self, X, y,
            sample_weight=None,
            optimiser="BGD",
            learning_rate=0.001,
            regularisation="ridge",
            lambda_reg=0,
            num_iterations=10000):
        """
        fit method that fits the parameters coef_ and intercept_ to the training set X
        """
        # no implementation of sample_weight attribute
        self.coef_ = np.zeros((X.shape[1],1))
        self.intercept_ = 0

        # Batch Gradient Descent
        m,n = X.shape
        costs = []
        for _ in range(num_iterations):
            J = 1/m * np.square(X@self.coef_ + self.intercept_ - y).sum()
            costs.append(J)
            grad_coef = 2/m * np.sum(X.T@(X@self.coef_ + self.intercept_ - y), axis=1, keepdims=True)
            grad_intercept = 2/m * np.sum((X@self.coef_ + self.intercept_ - y))
            self.coef_ -= learning_rate*grad_coef
            self.intercept_ -= learning_rate*grad_intercept

        return self

    def get_params(self, deep=True):
        """
        get parameters for this estimator
        """
        return dict([["normalise", self.normalise]])

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
            try:
                setattr(self, key, value)
            except Exception as ex:
                print(ex)
                print("Error while setting the parameters")
        return self

# for testing purposes
if __name__=="__main__":
    from sklearn import linear_model 
    lin_reg_cb = LinearRegression()
    lin_reg_sk = linear_model.LinearRegression()
    np.random.seed(16)
    X = np.arange(10,50,2, dtype="float64")
    X += np.random.randn(X.shape[0])
    X = X.reshape(X.shape[0],1)
    y = 2.5 * np.arange(10,50,2, dtype="float64") - 24
    y += np.random.randn(y.shape[0])
    y = y.reshape(y.shape[0],1)