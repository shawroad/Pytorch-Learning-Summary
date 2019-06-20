"""

@file   : 002-torch与numpy.py

@author : xiaolu

@time   : 2019-06-19

"""
import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)   # numpy2torch-tensor
tensor2array = torch_data.numpy()    # torch-tensor2numpy
print(torch_data)
print(tensor2array)
print("*"*100)


# 各种类型tensor
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # float32
print(np.abs(data))
print(torch.abs(tensor))
print(np.sin(data))
print(torch.sin(tensor))
print(np.mean(data))
print(torch.mean(tensor))
print("*"*100)


# 矩阵相乘
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
print(np.matmul(data, data))
print(torch.mm(tensor, tensor))
print("*"*100)
