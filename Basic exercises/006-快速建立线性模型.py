"""

@file   : 006-快速建立线性模型.py

@author : xiaolu

@time   : 2019-06-20

"""
import torch
import torch.nn.functional as F


# 一般定义模型的结构
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


net1 = Net(1, 10, 1)
print(net1)
print("*"*100)

# 直接定义 方便快捷
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
print(net2)
