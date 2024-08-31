import numpy as np
import sigmoid as sig

def predict(Theta1, Theta2, X):
    p = np.zeros((X.shape[0], 1))

    # add bias term to X
    bias = np.ones((X.shape[0], 1))
    a = np.hstack((bias, X))

    # compute first layer
    z_2 = a @ Theta1.T
    a_2 = sig.sigmoid(z_2)

    # add bias term
    bias = np.ones((a_2.shape[0], 1))
    a_2 = np.hstack((bias, a_2))

    # compute second layer
    z_3 = a_2 @ Theta2.T
    a_3 = sig.sigmoid(z_3)

    # a_3 is of size nrows(X), 3. Still need to OvA
    p = np.argmax(a_3, axis=1)[:, np.newaxis] + 1

    # h_x is now a_3
    h_x = a_3
    return p, h_x
