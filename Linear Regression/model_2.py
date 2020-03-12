"""
class define model with the help of PyTorch

Steps:
  1. design a model using Model class
  2. construct loss and optimizer through PyTorch API
  3. Traning: forward, backward and update

Note: when we design a model, we always override the two methods in PyTorch: __init__ and forward
"""

import torch
import time

# dataset
# Note: tensor only takes one-d
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class Model(torch.nn.Module):
  def __init__(self):
    # instantiate two nn.Linear modules
    super(Model, self).__init__()
    # input size (left): 1, and output value (right): 1 because each x_data and y_data has one val
    self.linear = torch.nn.Linear(1, 1)

  def forward(self, x):
    y_pred = self.linear(x)
    return y_pred

# create model obj which appliable to run the training between devices (CPU/ GPU)
model = Model().to()

criterion = torch.nn.MSELoss(size_average=False)

# use SGD as the optimizer. Inside the SGD, we pass the parameters that we want to update along with learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)

# This one not working
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)

# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)

# start the timer
start_time = time.time()

# training loop
for item in range(500):
  # foward propagation: compute predicted y by passing x to the model
  y_pred = model(x_data)

  # compute and output the loss
  loss = criterion(y_pred, y_data)
  print(item, loss.data)

  # reset the gradient
  optimizer.zero_grad()

  # perform backward propagation
  loss.backward()
  
  # update the gradient (weight w)
  optimizer.step()
  # Note: the w that we want to find is 2.0

# end the timer
end_time = time.time()

hour_var = torch.tensor([[4.0]])
print("\nPredict (after training)", 4, model.forward(hour_var).data[0][0])

# output the training timer
print("--- The model was trained in %s seconds ---" % (round(end_time - start_time, 4)))