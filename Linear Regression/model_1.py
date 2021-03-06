# linear regression (linear prediction) on # of hours study --> exam result
# supervised learning

import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0 # a random guess: random value
weight_list = []
mse_list = []

def forward(x):
    """
    forward passing
    INPUT: a float val
    OUTPUT: return (input * random val)
    """
    return x * w

def loss(x, y):
    """
    compute loss for the model
    INPUT: two float x and y
    OUTPUT: return (y_pred - y)^2
    """
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

for w in np.arange(0.0, 4.1, 0.1):
    print("w = ", w)
    loss_sum = 0

    for x_val, y_val in zip(x_data, y_data):
        # forward propagation
        y_pred_val = forward(x_val)

        # backward propagation
        L = loss(x_val, y_val)

        # sum the loss and then output it
        loss_sum += L
        print("\t", x_val, y_val, y_pred_val, L)

    # equation: 1/N ∑ (starts n=1 and to N) (ŷ_n - y_n)^2
    print("MSE = ", loss_sum / 3)
    weight_list.append(w)
    mse_list.append(loss_sum / 3)

plt.plot(weight_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()