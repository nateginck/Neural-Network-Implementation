import scipy
import numpy as np
import matplotlib.pyplot as plt
import predict
import nnCost
from sGD import sGD
import sigmoidGradient
from sklearn.metrics import accuracy_score

# import torch

import sigmoidGradient

# Check that GPU is detected for Pytorch
# print(torch.cuda.get_device_name(0))

# load matlab data
data = scipy.io.loadmat(r'input/HW7_Data2_full.mat')
weights = scipy.io.loadmat(r'input/HW7_weights_3_full.mat')

X = data['X']
y = data['y_labels']

data = np.concatenate((X, y), axis=1)

# 0a. pick 16 images and display with label

# shuffle data
np.random.shuffle(data)

# pick first 16 images to print
selected = data[:16]

# print 16 images with their label
fig, axs = plt.subplots(4, 4)
for i in range(16):
    ax = axs[i // 4, i % 4]
    img_label = selected[i, -1]

    img = np.reshape(selected[i, :-1], (32, 32)).T

    ax.imshow(img, cmap='gray')

    # convert label to string
    if img_label == 1: img_label = "airplane"
    if img_label == 2: img_label = "automobile"
    if img_label == 3: img_label = "truck"
    ax.set_title(f"{img_label}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(r'output/ps7-0-a-1.png')
plt.close()

# 0b. Randomly split data into testing and traing sets

# data is already shuffled before image selection
X_train = data[:13000, :-1]
y_train = data[:13000, -1]
X_test = data[13000:, :-1]
y_test = data[13000:, -1]

print(X_train.shape)

# 1b. Call predict function with weights
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

p, h_x = predict.predict(Theta1, Theta2, X)

# see accuracy
accuracy = accuracy_score(y, p)

print("Accuracy on entire dataset:", accuracy)

# 2b. Test nnCost function
J0 = nnCost.nnCost(Theta1, Theta2, X, y, 3, 0)
print("Cost with Lambda = 0:", J0)

J1 = nnCost.nnCost(Theta1, Theta2, X, y, 3, 1)
print("Cost with Lambda = 1:", J1)

J2 = nnCost.nnCost(Theta1, Theta2, X, y, 3, 2)
print("Cost with Lambda = 2:", J2)

# 3. Derivative of activation function
z = np.array([-10, 0, 10]).reshape(-1, 1)


g_prime = sigmoidGradient.sigmoidGradient(z)

print("The sigmoid gradient when z =\n", z)
print("Sigmoid Gradient:\n", g_prime)

# 4d. Pick learning rate alpha
alpha = 0.001
print("Alpha:", alpha)

# 4e. Call sGD on training set
# Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 1, alpha, 1)

# 5. Calculate and Print results
# MAXEPOCHS = 50
Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 0.1, alpha, 50)
p, h_x = predict.predict(Theta1, Theta2, X_train)
L01_train = accuracy_score(p, y_train)
p, h_x = predict.predict(Theta1, Theta2, X_test)
L01_test = accuracy_score(p, y_test)
cost01 = nnCost.nnCost(Theta1, Theta2, X_test, y_test, 3, 0.1)
cost01_train = nnCost.nnCost(Theta1, Theta2, X_train, y_train, 3, 0.1)

Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 1, alpha, 50)
p, h_x = predict.predict(Theta1, Theta2, X_train)
L1_train = accuracy_score(p, y_train)
p, h_x = predict.predict(Theta1, Theta2, X_test)
L1_test = accuracy_score(p, y_test)
cost1 = nnCost.nnCost(Theta1, Theta2, X_test, y_test, 3, 1)
cost1_train = nnCost.nnCost(Theta1, Theta2, X_train, y_train, 3, 1)


Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 2, alpha, 50)
p, h_x = predict.predict(Theta1, Theta2, X_train)
L2_train = accuracy_score(p, y_train)
p, h_x = predict.predict(Theta1, Theta2, X_test)
L2_test = accuracy_score(p, y_test)
cost2 = nnCost.nnCost(Theta1, Theta2, X_test, y_test, 3, 2)
cost2_train = nnCost.nnCost(Theta1, Theta2, X_train, y_train, 3, 2)



print("MaxEpochs = 50")
print("Lambda   Training accuracy,  testing accuracy, test cost, train cost")
print("0.1    ", L01_train, L01_test, "\t", cost01, cost01_train)
print("1      ", L1_train, L1_test, "\t", cost1, cost1_train)
print("2      ", L2_train, L2_test, "\t", cost2, cost2_train)

# MAXEPOCHS = 300
Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 0.1, alpha, 300)
p, h_x = predict.predict(Theta1, Theta2, X_train)
L01_train = accuracy_score(p, y_train)
p, h_x = predict.predict(Theta1, Theta2, X_test)
L01_test = accuracy_score(p, y_test)
cost01 = nnCost.nnCost(Theta1, Theta2, X_test, y_test, 3, 0.1)
cost01_train = nnCost.nnCost(Theta1, Theta2, X_train, y_train, 3, 0.1)


Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 1, alpha, 300)
p, h_x = predict.predict(Theta1, Theta2, X_train)
L1_train = accuracy_score(p, y_train)
p, h_x = predict.predict(Theta1, Theta2, X_test)
L1_test = accuracy_score(p, y_test)
cost1 = nnCost.nnCost(Theta1, Theta2, X_test, y_test, 3, 1)
cost1_train = nnCost.nnCost(Theta1, Theta2, X_train, y_train, 3, 1)


Theta1, Theta2 = sGD(1024, 3, 3, X_train, y_train, 2, alpha, 300)
p, h_x = predict.predict(Theta1, Theta2, X_train)
L2_train = accuracy_score(p, y_train)
p, h_x = predict.predict(Theta1, Theta2, X_test)
L2_test = accuracy_score(p, y_test)
cost2 = nnCost.nnCost(Theta1, Theta2, X_test, y_test, 3, 2)
cost2_train = nnCost.nnCost(Theta1, Theta2, X_train, y_train, 3, 2)


print("MaxEpochs = 300")
print("Lambda   Training accuracy,  testing accuracy, test cost, train cost")
print("0.1    ", L01_train, L01_test, "\t", cost01, cost01_train)
print("1      ", L1_train, L1_test, "\t", cost1, cost1_train)
print("2      ", L2_train, L2_test, "\t", cost2, cost2_train)

