import numpy as np
import sigmoid
from sigmoidGradient import *
from nnCost import nnCost
import matplotlib.pyplot as plt

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, Lambda, alpha, MaxEpochs):
    # 4a. Define Theta1 and Theta2
    Theta1 = np.random.uniform(-0.18, 0.18, (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(-0.18, 0.18, (num_labels, hidden_layer_size + 1))

    # recode y
    y_recode = np.zeros((X_train.shape[0], num_labels))

    # shift values of y so index at 0
    y_new = y_train.astype(int) - 1
    y_recode[np.arange(X_train.shape[0]), y_new.flatten()] = 1

    # Loop over each Epoch and perform backpropagation and gradient descent
    # one training example at time

    # costs = np.zeros(X_train.shape[0]) # used for creating plot in 4e.

    for EPOCH in range(MaxEpochs):

    # compute gradient for cost function, and use gradient descent to update weights
        for q in range(X_train.shape[0]):
            # first run forward pass to compute activations

            # add bias
            a1 = np.hstack([np.ones(1), X_train[q]])
            z2 = Theta1 @ a1
            g2 = sigmoid.sigmoid(z2)

            # add bias
            a2 = np.hstack([1, g2])
            z3 = Theta2 @ a2
            a3 = sigmoid.sigmoid(z3)

            # calculate error
            delta3 = a3 - y_recode[q, :]

            # backpropagation
            delta2 = Theta2.T[1:] @ delta3 * sigmoidGradient(z2)

            # update gradients, use np.outer for more efficient calculation
            D2 = np.outer(delta3, a2)
            D1 = np.outer(delta2, a1)

            # add regularization
            R2 = Lambda / X_train.shape[0] * np.hstack([np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]])
            R1 = Lambda / X_train.shape[0] * np.hstack([np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]])

            # Update Weight Matrices
            Theta1 = Theta1 - alpha * (D1 + R1)
            Theta2 = Theta2 - alpha * (D2 + R2)

            # remove after part 4
            # costs[q] = nnCost(Theta1, Theta2, X_train, y_train, 3, 1)

    # plot used in part 4
    # plt.plot(costs, label='Cost')
    # plt.title('Cost over Back Propagation')
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.savefig(r'output/ps7-4-e-1.png')
    # plt.close()

    return Theta1, Theta2
