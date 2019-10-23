"""

@file  : 006-mnist数据集分类--内涵数据集的构造.py

@author: xiaolu

@time  : 2019-10-22

"""
from keras.datasets import mnist
from keras.utils import to_categorical
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


# 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)    # 28-5+1 = 24
        self.conv2 = nn.Conv2d(6, 16, 5)   # 上一步还进行的池化 12   12-5+1=8
        self.fc1 = nn.Linear(16*4*4, 120)  # 上一步再进行池化  4  应该是16x4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)  # 将数据压直
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 写一个继承类 将数据整理成一个很好的样子  然后放进DataLoader()中去
class DataTxt(Dataset):
    def __init__(self):
        self.Data = x_train
        self.Labels = y_train

    def __getitem__(self, item):
        data = torch.from_numpy(self.Data[item])
        label = torch.from_numpy(self.Labels[item])
        return data, label

    def __len__(self):
        return self.Data.shape[0]

import numpy as np
if __name__ == '__main__':
    # 我们加载keras中的内置数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)
    x_test = x_test.reshape((-1, 1, 28, 28)) / 255.
    x_test = x_test.astype(np.float32)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    dxt = DataTxt()
    trainloader = torch.utils.data.DataLoader(
        dxt,
        batch_size=128,
        shuffle=True,
    )

    dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # print(images)
    # print(labels)

    net = Net()
    criterion = nn.MultiLabelSoftMarginLoss()  # 多标签损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 输入数据
            inputs, labels = data
            # print(inputs.size())   # torch.Size([32, 1, 28, 28])
            # print(labels.size())   # torch.Size([32, 10])

            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += loss.data[0]
            print('epoch: %d, step: %d, loss: %f' % (epoch, i, loss))
