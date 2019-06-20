"""

@file   : 004-regression.py

@author : xiaolu

@time   : 2019-06-20

"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import optim

torch.manual_seed(1)    # reproducible  设置随机种子

# unsqueeze相当于将数据扩充维度 相当于keras中的expand_dim()
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)   # unsqueeze相当于把一维数据变二维 再加个括号， shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())     # 计算一个真实的y  加了点噪声  y data (tensor), shape=(100, 1)

# 因为torch仅仅能训练Variable 所以我们需要转换x,y到Variable
x, y = Variable(x), Variable(y)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)    # 隐藏层
        self.predict = torch.nn.Linear(n_hidden, n_output)     # 输出层

    def forward(self, x):
        x = F.relu(self.hidden(x))    # 激活隐层
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)   # 实例化网络
# print(net)    # 打印网络结构


# 定义优化器和损失函数
optimizer = optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()    # 均方损失


# 应用阻塞模式进行绘图
# 如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。
# 要想防止这种情况，需要在plt.show()之前加上ioff()命令。
plt.ion()
for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()   # 所有参数的梯度设为零 开始反向传递
    loss.backward()   # 反向传播
    optimizer.step()     # 应用梯度更新

    if t % 5 == 0:
        # 每五步打印一次
        plt.cla()    # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
        plt.scatter(x.data.numpy(), y.data.numpy())   # 原始的点
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)    # 预测
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})   # 画损失
        plt.pause(0.1)   # 每个0.1进行变动一次


plt.ioff()
plt.show()
