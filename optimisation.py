import numpy as np

class BGDLinearRegression:

    """Compute Batch Gradient Descent algorithm
    to find the optimal values for linear regression.
    """

    def __init__(
            self, X, y, coef, intercept,
            learning_rate, regularisation,
            lambda_reg, num_iterations):
        # Initialise parameters
        self.X = X
        self.y = y
        self.coef = coef
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations
        # Keep costs
        self.costs = []
    
    def __repr__(self):
        return "Batch Gradient Descent for Linear Regression (" + \
            f"learning_rate={self.learning_rate}, " + \
            f"regularisation={self.regularisation}, " + \
            f"lambda_reg={self.lambda_reg}, " + \
            f"num_iterations={self.num_iterations})"
            
    def batch_gradient_descent(self):
        """Compute the gradient descent on the parameters of
        the model
        """
        m,n = self.X.shape
        for _ in range(self.num_iterations):
            predictions = self.X@self.coef + self.intercept
            J = 1/m * np.square(predictions - self.y).sum()
            self.costs.append(J)
            grad_coef = 2/m * self.X.T @ (predictions-self.y)
            grad_intercept = 2/m * np.sum(predictions - self.y)
            self.coef -= self.learning_rate*grad_coef
            self.intercept -= self.learning_rate*grad_intercept
        return self


class BGDLearningRateOptimiser:

    """Find the best learning rate for Batch Gradient Descent
    to converge as fast as possible without diverging.
    """

    def __init__(
            self):
        pass