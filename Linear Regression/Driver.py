import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0 # a random guess: random value
weight_list = []
mse_list = []

# model for the forward passing
def forward(x):
    return x * w

# loss func
def loss(x, y):
    # equation: (x * w - y)^2
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

for w in np.arange(0.0, 4.1, 0.1):
    print("w = ", w)
    loss_sum = 0

    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        L = loss(x_val, y_val)
        loss_sum += L
        print("\t", x_val, y_val, y_pred_val, L)

    print("MSE = ", loss_sum / 3)
    weight_list.append(w)
    mse_list.append(loss_sum / 3)

plt.plot(weight_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()