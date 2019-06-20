"""

@file   : 010-CNN.py

@author : xiaolu

@time   : 2019-06-20

"""
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim
import matplotlib as mp
from matplotlib import cm
from sklearn.manifold import TSNE

torch.manual_seed(1)

# 一些参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False


# 加载数据集  如果不存在  则进行下载
if not(os.path.exists('./data/')) or not os.listdir('./data/'):
    DOWNLOAD_MNIST = True


# 对数据进行处理  将图片转换为tensor  并减去均值除以方差(标准化)
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

# 加载数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# print(train_dataset.data.size())    # torch.Size([60000, 28, 28])
# print(train_dataset.targets.size())    # torch.Size([10000])

# # 画出前九张图
# for i in range(1, 10):
#     plt.subplot('33{}'.format(i))
#     plt.imshow(train_dataset.data[i], cmap='gray')
#     plt.title('%i' % train_dataset.targets[i])
# plt.subplots_adjust(left=1, bottom=1, right=3, top=3, wspace=5, hspace=5)
# plt.show()

# 加载测试集  我们只选取2000张  并将shape(2000, 28, 28)转为 shape=(2000, 1, 28, 28)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=DOWNLOAD_MNIST)
test_x = torch.unsqueeze(test_dataset.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_dataset.targets[:2000]
print(test_x.size())
print(test_y.size())


# 定义模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # input shape (1, 28, 28)
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(     # input shape (16, 14, 14)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 相当于faltten作用  # (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x     # 返回x为了后面的可视化


cnn = CNN()
# print(cnn)   # 查看一下模型结构

optimizer = optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Visualize last layer")
    plt.show()
    plt.pause(0.01)


plt.ion()

# 训练并测试
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]  # 得到输出
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # 每50步测试一下 看一下最后一层的样子
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print("Epoch:{}, train_loss:{}, test_accuracy:{}".format(epoch, loss.data.numpy(), accuracy))

            # 可视化最后一层
            # TSNE 是降维
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            labels = test_y.numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels)


plt.ioff()


# 取测试集前10个进行预测
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print("真实标签:", test_y[:10].numpy())
print("预测标签:", pred_y)

