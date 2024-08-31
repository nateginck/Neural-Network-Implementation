# signmoid function

import numpy as np

def sigmoid(z):
    # compute 1/(1 + e^-z
    return 1 / (1 + np.exp(-z))


