import sigmoid
# function to compute gradient of sigmoid function
def sigmoidGradient(z):
    return sigmoid.sigmoid(z) * (1 - sigmoid.sigmoid(z))

