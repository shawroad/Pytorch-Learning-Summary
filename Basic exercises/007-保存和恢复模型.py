"""

@file   : 007-保存和恢复模型.py

@author : xiaolu

@time   : 2019-06-20

"""
import torch
import matplotlib.pyplot as plt
from torch import optim


def build_model_save(x, y):
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = optim.SGD(net1.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(200):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # 两种方式保存模型
    torch.save(net1, 'net.pkl')  # 保存完整模型
    torch.save(net1.state_dict(), 'net_params.pkl')   # 只保留参数


def restore_net(x, y):
    # 恢复第一种方法保存模型
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    print(prediction)

    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params(x, y):
    # 恢复第二种方法保存的模型
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible
    # 制造一些数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # 加写噪声算真实y shape=(100, 1)

    build_model_save(x, y)
    restore_net(x, y)
    restore_params(x, y)
