"""

@file   : 003-variable使用.py

@author : xiaolu

@time   : 2019-06-19

"""
import torch
from torch.autograd import Variable


# Variable in torch is to build a computational graph
tensor = torch.FloatTensor([[1, 2], [3, 4]])  # 建立tensor
variable = Variable(tensor, requires_grad=True)   # 建立能求导的变量
print(tensor)
print(variable)
print("*"*100)

# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)
print("*"*100)

v_out.backward()   # variable可以反向传播
print(variable.grad)
# t_out.backward()  # 将会报错  因为不是variable


print("*"*100)
v_out = 1/4 * sum(variable*variable)   # 以上面的均值等价
# d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)
print("*"*100)


print(variable)
print(variable.data)
print(variable.data.numpy())
