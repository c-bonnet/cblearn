import numpy as np

class LinearRegression:

    """Estimator that can compute linear regression."""

    def __init__(
            self, normalise=False, optimiser="BGD",
            learning_rate=None, regularisation="ridge",
            lambda_reg=0, num_iterations=1000,
            ):
        """Initialise LinearRegression object,
        set normalise to True to normalise the input X
        """
        # Intialise parameters
        self.normalise=normalise
        self.optimiser=optimiser
        self.learning_rate=learning_rate
        self.regularisation=regularisation
        self.lambda_reg=lambda_reg
        self.num_iterations=num_iterations
        # Initialise model parameters
        self.coef = np.zeros((1, 1))
        self.intercept = 0
        # Costs savings
        self.costs = []
    
    def __repr__(self):
        """Copy of scikit-learn representation of a LinearRegression
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
        """Fit method that fits the parameters coef and intercept
        to the training set X
        """
        (m, n) = X.shape
        self.coef = np.zeros((n, 1))
        self.intercept = 0

        # Batch Gradient Descent
        if self.optimiser == "BGD":
            # learning_rate optimisation
            # If not given, try to find the best
            if self.learning_rate is None:
                self._optimise_learning_rate(X, y)
            # Compute Batch Gradient Descent on the training set X
            self._batch_gradient_descent(X, y, self.num_iterations)
        
        return self

    def predict(self, X):
        """Predict using the linear model"""
        try:
            predictions = X@self.coef + self.intercept
        except Exception as ex:
            print("Error while predicting")
            print(ex)
            print("Thus predicting 0s instead...")
            predictions = np.zeros(X.shape[0])
        return predictions

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of
        the prediction
        """
        #TODO
        pass

    def set_params(self, **params):
        """Set the parameters of this estimator"""
        for key, value in params.items():
            try:
                setattr(self, key, value)
            except Exception as ex:
                print(ex)
                print("Error while setting the parameters")
        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "normalise": self.normalise,
            }
    
    def get_model_params(self):
        """Return model parameters"""
        return {
            "coef": self.coef,
            "intercept": self.intercept,
            }

    def _optimise_learning_rate(
            self, X, y,
            min_low_bound=0,
            max_upp_bound=10,
            poly_iterations=20,
            poly_parts=5,
            rel_threshold=0.01
            ):
        """Find the best learning rate and update it for
        Batch Gradient Descent to converge as fast as possible
        without diverging.
        """
        lower_bound, upper_bound = min_low_bound, max_upp_bound
        # Polychotomy algorithm to find the best learning_rate in 
        # between lower_bound and upper_bound
        for _ in range(poly_iterations):
            poly_costs = np.zeros(poly_parts + 1)
            for poly_part in range(poly_parts + 1):
                self.learning_rate = (
                    lower_bound
                    + poly_part/poly_parts * (upper_bound-lower_bound)
                    )
                poly_costs[poly_part] = self.__costs_diff(X, y)
            poly_costs = np.nan_to_num(poly_costs, nan=np.inf)
            min_part = np.argmin(poly_costs)
            low_part = max(min_part - 1, 0)
            upp_part = min(min_part + 1, poly_parts)
            new_lower_bound = (
                lower_bound
                + low_part/poly_parts * (upper_bound-lower_bound)
                )
            new_upper_bound = (
                lower_bound
                + upp_part/poly_parts * (upper_bound-lower_bound)
                )
            lower_bound, upper_bound = new_lower_bound, new_upper_bound
            relative_diff = (upper_bound-lower_bound) / lower_bound
            if relative_diff < rel_threshold:
                break
        self.learning_rate = lower_bound
        return self

    def _batch_gradient_descent(
            self, X, y,
            num_iterations,
            ):
        """Compute the gradient descent to optimise the parameters of
        the model.
        """
        (m, n) = X.shape
        for _ in range(num_iterations):
            predictions = X@self.coef + self.intercept
            J = 1/m * np.square(predictions - y).sum()
            self.costs.append(J)
            grad_coef = 2/m * X.T @ (predictions-y)
            grad_intercept = 2/m * np.sum(predictions - y)
            self.coef -= self.learning_rate*grad_coef
            self.intercept -= self.learning_rate*grad_intercept
        return self

    def _reset_model_params(self, reset_costs=True):
        """Reset model parameters."""
        self.coef = np.zeros_like(self.coef)
        self.intercept = 0
        if reset_costs:
            self.costs = []
        return self

    def __costs_diff(self, X, y, cost_num_iterations=5):
        """Compute Batch Gradient Descent for a couple of iterations
        to check whether the algorithm is diverging or not.
        """
        self._reset_model_params(reset_costs=True)
        self._batch_gradient_descent(X, y, num_iterations=cost_num_iterations)
        # Compute the difference of costs after some iterations of
        # Batch Gradient Descent.
        costs_diff = self.costs[-1] - self.costs[0]
        return costs_diff


