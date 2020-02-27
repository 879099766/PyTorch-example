import torch
import timeit

x = torch.rand(6, 4, 7)
z = torch.rand(6, 4, 7)

# print(x)
# print("\n////////////////////////////////////////\n")
# y = torch.rand(2,3)
# h = torch.rand(2,3)
#
# print("Tensor y: %s \n" % (y))
# print("Tensor h: %s \n" % (h))

num = 100
range_time = [];
temp = 0

for x in range(num):
    start = timeit.default_timer()
    c = torch.add(x, z)
    end = timeit.default_timer()
    range_time.append(end)
    # print(c)
    # print("The run time for adding two tensors is: " , end)

for x in range(num):
    temp += range_time[x]

print(temp)
print(temp/num)