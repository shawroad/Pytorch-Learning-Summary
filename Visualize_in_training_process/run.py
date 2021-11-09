"""
@file   : run.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-11-08
"""
import torch
import torchvision
from torch import nn
from model import ConvNet
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

'''
安装:
pip install tensorboardX
pip install tensorboard

代码运行后，在和run在同一个目录下的命令行上输入:
tensorboard --logdir="./data/log"
即可看到准确率和损失的变化
'''

if __name__ == '__main__':
    MyConvNet = ConvNet()

    # 准备训练用的MNIST数据集
    train_data = torchvision.datasets.MNIST(
        root="./data/MNIST",  # 提取数据的路径
        train=True,  # 使用MNIST内的训练数据
        transform=torchvision.transforms.ToTensor(),  # 转换成torch.tensor
        download=False  # 如果是第一次运行的话，置为True，表示下载数据集到root目录
    )

    # 定义loader
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    test_data = torchvision.datasets.MNIST(
        root="./data/MNIST",
        train=False,  # 使用测试数据
        download=False
    )

    # 将测试数据压缩到0-1
    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    test_data_x = torch.unsqueeze(test_data_x, dim=1)
    test_data_y = test_data.targets

    # 打印一下测试数据和训练数据的shape
    print("test_data_x.shape:", test_data_x.shape)   # torch.Size([10000, 1, 28, 28])
    print("test_data_y.shape:", test_data_y.shape)   # torch.Size([10000])

    logger = SummaryWriter(log_dir="data/log")

    # 获取优化器和损失函数
    optimizer = torch.optim.Adam(MyConvNet.parameters(), lr=3e-4)
    loss_func = nn.CrossEntropyLoss()
    log_step_interval = 100  # 记录的步数间隔

    for epoch in range(5):
        print("epoch:", epoch)
        # 每一轮都遍历一遍数据加载器
        for step, (x, y) in enumerate(train_loader):
            # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
            predict = MyConvNet(x)
            loss = loss_func(predict, y)
            optimizer.zero_grad()  # 清空梯度（可以不写）
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新网络
            global_iter_num = epoch * len(train_loader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
            if global_iter_num % log_step_interval == 0:
                # 控制台输出一下
                print("global_step:{}, loss:{:.2}".format(global_iter_num, loss.item()))
                # 添加的第一条日志：损失函数-全局迭代次数
                logger.add_scalar("train loss", loss.item(), global_step=global_iter_num)
                # 在测试集上预测并计算正确率
                test_predict = MyConvNet(test_data_x)
                _, predict_idx = torch.max(test_predict, 1)  # 计算softmax后的最大值的索引，即预测结果
                acc = accuracy_score(test_data_y, predict_idx)
                # 添加第二条日志：正确率-全局迭代次数
                logger.add_scalar("test accuary", acc.item(), global_step=global_iter_num)
                # 添加第三条日志：网络中的参数分布直方图
                for name, param in MyConvNet.named_parameters():
                    logger.add_histogram(name, param.data.numpy(), global_step=global_iter_num)