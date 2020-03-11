# a simple gradient descent through PyTorch
# continue with prediction on #of hours studies vs score result

# training data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# random guess
w = 1.0

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

def gradient(x, y):
    """
    compute the gradient of the graph
    INPUT: float val x and y
    OUTPUT: return the gradient of the graph using derivatives chain rules
    """
    return 2 * x * (x * w - y)

print("predict (before training)", 4, forward(4))

# training model
for item in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        # note: "0.01" is running rate
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss (x_val, y_val)

    print("progress:", item, "w = ", w, " loss = ", l)

# note: whatever the w is towards, that's the true val
print("predict (after training)", "4 hours", forward(4))