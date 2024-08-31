import predict
import numpy as np

# 2a. nnCost function
def nnCost(Theta1, Theta2, X, y, K, Lambda):
    # calculate h_x
    p, h_x = predict.predict(Theta1, Theta2, X)

    # recode y
    y_recode = np.zeros((X.shape[0], K))

    # shift values of y so index at 0
    y_new = y.astype(int) - 1
    y_recode[np.arange(X.shape[0]), y_new.flatten()] = 1

    # calculate cost
    J = -1/X.shape[0] * np.sum(y_recode * np.log(h_x) + (1 - y_recode) * np.log(1 - h_x))

    # add regularization term
    J += Lambda/(2 * X.shape[0]) * ((np.sum(Theta1[:, 1:] ** 2)) + np.sum(Theta2[:, 1:] ** 2))
    return J


