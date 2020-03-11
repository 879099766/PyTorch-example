# autograd from PyTorch

import torch

# training data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# random val
# Note: "w = Variable(torch.Tensor([1.0]),  requires_grad=True)" on the tutorial is deprecated
# below method is encouraged
w = torch.tensor([1.0], requires_grad=True)

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


# training model
for item in range(10):
    for x_val, y_val in zip(x_data, y_data):
        L = loss(x_val, y_val)
        # backward propagation in order to compute all gradients
        L.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])

        # update weight val
        # note: "0.01" is running rate
        w.data = w.data - 0.01 * w.grad.data

        # reset the gradient to zero after updating the weights
        w.grad.data.zero_()

    print("progress:", item, L.data[0])

# note: whatever the w is towards, that's the true val
print("predict (after training)", "4 hours", forward(4).data[0])