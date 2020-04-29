import numpy as np

from optimisation import BGDLinearRegression, BGDLearningRateOptimiser
import test_linear_model


class LinearRegression:

    """
    Estimator that can compute linear regression
    """

    def __init__(
            self, normalise=False, optimiser="BGD",
            learning_rate=None, regularisation="ridge",
            lambda_reg=0, num_iterations=1000):
        """
        initialise LinearRegression object
        set normalise to True to normalise the input X
        """
        # intialise parameters
        self.normalise=normalise
        self.optimiser=optimiser
        self.learning_rate=learning_rate
        self.regularisation=regularisation
        self.lambda_reg=lambda_reg
        self.num_iterations=num_iterations
        # initialise model parameters
        self.coef_ = np.empty((1))
        self.intercept_ = 0
        # costs savings
        self.costs_ = []
    
    def __repr__(self):
        """
        copy of scikit-learn representation of a LinearRegression
        estimator
        """
        return  "LinearRegression(" +\
                f"normalise={self.normalise}, " +\
                f"optimiser={self.optimiser}, " +\
                f"learning_rate={self.learning_rate}, " +\
                f"regularisation={self.regularisation}, " +\
                f"lambda_reg={self.lambda_reg}, " +\
                f"num_iterations={self.num_iterations})"
    
    def fit(self, X, y):
        """
        fit method that fits the parameters coef_ and intercept_
        to the training set X
        """
        m,n = X.shape
        self.coef_ = np.zeros((n,1))
        self.intercept_ = 0

        # Batch Gradient Descent
        if self.optimiser == "BGD":
            # learning_rate optimisation
            # if not given, try to find the best
            if self.learning_rate is None:
                lr_optimiser = BGDLearningRateOptimiser(
                    X, y, self.coef_, self.intercept_,
                    self.learning_rate, self.regularisation,
                    self.lambda_reg)
                lr_optimiser.optimise()
                self.learning_rate = lr_optimiser.learning_rate
            # comute Batch Gradient Descent on the training set X
            bgd = BGDLinearRegression(
                X, y, self.coef_, self.intercept_,
                self.learning_rate, self.regularisation,
                self.lambda_reg, self.num_iterations)
            bgd.batch_gradient_descent()
            self.coef_ = bgd.coef
            self.intercept_ = bgd.intercept
            self.costs_ = bgd.costs
        
        return self

    def get_params(self, deep=True):
        """
        get parameters for this estimator
        """
        return {
            "normalise": self.normalise
            }

    def predict(self, X):
        """
        predict using the linear model
        """
        try:
            predictions = X@self.coef_ + self.intercept_
        except Exception as ex:
            print(ex)
            print("Error while predicting")
            print("Thus predicting 0s instead...")
            predictions = np.zeros(X.shape[0])
        return predictions


    def score(self, X, y, sample_weight=None):
        """
        return the coefficient of determination R^2 of the prediction
        """
        #TODO
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
    
    def get_model_params(self):
        """
        return model parameters
        """
        return {
            "coef": self.coef,
            "intercept": self.intercept_
            }

# for testing purposes
if __name__=="__main__":
    test_linear_model.run()
