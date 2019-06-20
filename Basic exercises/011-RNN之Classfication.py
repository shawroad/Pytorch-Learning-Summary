"""

@file   : 011-RNN之Classfication.py

@author : xiaolu

@time   : 2019-06-20

"""
import torch
from torch import nn
from torchvision import datasets
import torch.utils.data as Data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch import optim

torch.manual_seed(1)

# 一些参数
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True


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
# print(train_dataset.targets.size())    # torch.Size([60000])

# # 画出前九张图
# for i in range(1, 10):
#     plt.subplot('33{}'.format(i))
#     plt.imshow(train_dataset.data[i], cmap='gray')
#     plt.title('%i' % train_dataset.targets[i])
# plt.subplots_adjust(left=1, bottom=1, right=3, top=3, wspace=5, hspace=5)
# plt.show()

# 加载测试集  我们只选取2000张  并将shape(2000, 28, 28)转为 shape=(2000, 1, 28, 28)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=DOWNLOAD_MNIST)
test_x = test_dataset.data.type(torch.FloatTensor)[:2000] / 255.
test_y = test_dataset.targets[:2000]
print(test_x.size())   # torch.Size([2000, 28, 28])
print(test_y.size())   # torch.Size([2000])


# 定义RNN模型结构
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,    # 隐层的输出
            num_layers=1,    # 可以把LSTM搞成多层堆叠
            batch_first=True,  # input 和 output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)   # 每步都有输出
        # h_n shape (n_layers, batch, hidden_size)    # 每层LSTM最后的输出
        # h_c shape (n_layers, batch, hidden_size)    # 每层LSTM最后的细胞状态
        r_out, (h_n, h_c) = self.rnn(x, None)     # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = optim.Adam(rnn.parameters(), lr=LR)
loss_fun = nn.CrossEntropyLoss()

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # view()的作用跟reshape()一样
        b_x = b_x.view(-1, 28, 28)
        output = rnn(b_x)
        loss = loss_fun(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)    # (samples, time_step, input_size)
            pred_y = torch.max(test_output, dim=1)[1].data.numpy()
            # print(pred_y.shape)    # (2000, )
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch:{}, train_loss:{}, test_accuracy:{}'.format(epoch, loss.data.numpy(), accuracy))


# 从测试集选10个进行预测
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print("真实标签:", test_y[:10].numpy())
print("预测标签:", pred_y)
