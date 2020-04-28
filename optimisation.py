import numpy as np

class BGDLinearRegression:
    """
    compute Batch Gradient Descent algorithm to find the optimal values for linear regression
    """
    def __init__(self, X, y, coef, intercept, learning_rate, regularisation,
        lambda_reg, num_iterations):
        # initialise parameters
        self.X = X
        self.y = y
        self.coef = coef
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations
        # keep costs
        self.costs = []
    
    def __repr__(self):
        return "Batch Gradient Descent for Linear Regression (" + \
            f"learning_rate={self.learning_rate}, " + \
            f"regularisation={self.regularisation}, " + \
            f"lambda_reg={self.lambda_reg}, " + \
            f"num_iterations={self.num_iterations})"
            
    def batch_gradient_descent(self):
        """
        compute the gradient descent on the parameters of the model
        """
        m,n = self.X.shape
        for _ in range(self.num_iterations):
            grad_coef = 2/m * self.X.T@(self.X@self.coef + self.intercept - self.y)
            grad_intercept = 2/m * np.sum(self.X@self.coef + self.intercept - self.y)
            self.coef -= self.learning_rate*grad_coef
            self.intercept -= self.learning_rate*grad_intercept
            J = 1/m * np.square(self.X@self.coef + self.intercept - self.y).sum()
            self.costs.append(J)
        return self

class 