import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model as sk_lm 

import linear_model as cb_lm

def run():
    """Run some tests on the LinearRegression model"""
    # Make some dummy data
    np.random.seed(16)
    X = np.arange(10,150,2, dtype="float64")
    X += np.random.randn(X.shape[0])
    X = X.reshape(X.shape[0],1)
    y = 2.5*np.arange(10,150,2, dtype="float64") - 24
    y += np.random.randn(y.shape[0])
    y = y.reshape(y.shape[0],1)
    # Compute linear regression
    scores = [cb_lm.LinearRegression(
        num_iterations=100, learning_rate=lr/100000 \
        ).fit(X,y).costs for lr in range(100)]
    # Plot chart
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('hsv')
    for idx, lr in enumerate(range(1,10)):
        plt.plot(
            scores[lr], color=palette(10*idx),
            linewidth=1, alpha=0.9,
            label=f"learning_rate {lr/100000}")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Gradient descent with different learning rates")
    plt.legend()
    plt.show()

def get_sets():
    np.random.seed(16)
    X = np.arange(10,150,2, dtype="float64")
    X += np.random.randn(X.shape[0])
    X = X.reshape(X.shape[0],1)
    y = 2.5*np.arange(10,150,2, dtype="float64") - 24
    y += np.random.randn(y.shape[0])
    y = y.reshape(y.shape[0],1)
    return (X, y)

def test_learning_rate():
    X, y = get_sets()
    lin_reg = cb_lm.LinearRegression()
    lin_reg.fit(X,y)
    return X, y, lin_reg

def test_diabetes(regularisation=False, multivalue=True):
    (X, y) = datasets.load_diabetes(return_X_y=True)
    y_2 = np.concatenate((y.reshape(-1,1), y.reshape(-1,1)), axis=1)
    if not regularisation:
        lin_reg_cb = cb_lm.LinearRegression(num_iterations=200000)
    if regularisation:
        lin_reg_cb = cb_lm.LinearRegression(
            lambda_reg=1 ,num_iterations=200000
            )
    lin_reg_cb.fit(X, y)
    print("coef")
    print(lin_reg_cb.coef.T)
    print("intercept")
    print(lin_reg_cb.intercept)
    if not regularisation:
        lin_reg_sk = sk_lm.LinearRegression()
    if regularisation:
        lin_reg_sk = sk_lm.Ridge(alpha=1)
    lin_reg_sk.fit(X, y)
    print("coef")
    print(lin_reg_sk.coef_)
    print("intercept")
    print(lin_reg_sk.intercept_)
    # Multivalue regression
    if multivalue:
        if not regularisation:
            lin_reg_cb = cb_lm.LinearRegression(num_iterations=400000)
        if regularisation:
            lin_reg_cb = cb_lm.LinearRegression(
                lambda_reg=1 ,num_iterations=400000
                )
        lin_reg_cb.fit(X, y_2)
        print("coef")
        print(lin_reg_cb.coef.T)
        print("intercept")
        print(lin_reg_cb.intercept)
        if not regularisation:
            lin_reg_sk = sk_lm.LinearRegression()
        if regularisation:
            lin_reg_sk = sk_lm.Ridge(alpha=1)
        lin_reg_sk.fit(X, y_2)
        print("coef")
        print(lin_reg_sk.coef_)
        print("intercept")
        print(lin_reg_sk.intercept_)
        

# For testing purposes
# if __name__=="__main__":
#     run()