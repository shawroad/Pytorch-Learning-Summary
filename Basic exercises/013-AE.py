"""

@file   : 013-AE.py

@author : xiaolu

@time   : 2019-06-21

"""
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torch import optim


torch.manual_seed(43)

# 一些参数定义
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DownLoad = False

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)


train_data = datasets.MNIST(
    root='./data/',
    train=True,
    transform=transforms,
    download=DownLoad
)

# 看看读进来的数据集
# print(train_data.data.size())   # torch.Size([60000, 28, 28])
# print(train_data.targets.size())   # torch.Size([60000])
# plt.imshow(train_data.data[1].numpy(), cmap='gray')
# plt.title('{}'.format(train_data.targets[1]))
# plt.show()


# 再入数据 并一批一批的整
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# 定义模型结构
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),     # 压缩成三个特征
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 28*28),
            nn.Sigmoid(),      # 将特征压缩到(0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = optim.Adam(autoencoder.parameters(), lr=LR)

loss_fun = nn.MSELoss()

N_Test_Img = 5   # 测试5张图
f, a = plt.subplots(2, N_Test_Img, figsize=(5, 2))    # 只训练前五图片

plt.ion()
view_data = train_data.data[:N_Test_Img].view(-1, 28*28).type(torch.FloatTensor) / 255.
for i in range(N_Test_Img):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28)    # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28*28)    # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_fun(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Epoch:{}, train_loss:{}".format(epoch, loss.data.numpy()))
            _, decoded_data = autoencoder(view_data)
            for i in range(N_Test_Img):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()


# 可视化3D图  把中间那些三维数据画出来
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor) / 255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[: 200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
