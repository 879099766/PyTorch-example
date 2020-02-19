import torch

x = torch.empty(5, 3, 4)
print(x)
print("\n////////////////////////////////////////\n")
y = torch.rand(2,3)
h = torch.rand(2,3)

print("Tensor y: %s \n" % (y))
print("Tensor h: %s \n" % (h))

c = torch.add(y, h)
print(c)