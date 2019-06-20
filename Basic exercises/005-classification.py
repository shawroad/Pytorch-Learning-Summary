"""

@file   : 005-classification.py

@author : xiaolu

@time   : 2019-06-20

"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable


torch.manual_seed(43)

n_data = torch.ones(100, 2)   # shape=(100, 2)
# print(n_data.size())
# print(n_data)


x0 = torch.normal(2*n_data, 1)  # shape=(100, 2)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)  # shape=(100, 2)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)   # shape=(200, 2)  然后转为Float32类型
y = torch.cat((y0, y1),).type(torch.LongTensor)  # shape=(200,) 然后转为Longint类型
# print(x.size())
# print(y.size())
# print(x)
# print(y)


# # The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
# print(net)    # 打印网络结构


optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()     # 交叉损失熵

plt.ion()
for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        prediction = torch.max(out, dim=1)[1]    # 看输出那个维度大  dim是看从哪一维度统计最大值，最后面[1]是只取索引
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.cla()   # 清理动态变化的图
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
