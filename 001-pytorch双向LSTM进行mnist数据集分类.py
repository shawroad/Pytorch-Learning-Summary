"""

@file   : 001-pytorch双向LSTM进行mnist数据集分类.py

@author : xiaolu

@time   : 2019-06-12

"""
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义参数
sequence_length = 28    # 序列长度
input_size = 28   # 输入维度
hidden_size = 128  # 隐层输出维度
num_layers = 2   # 两层LSTM
num_classes = 10   # 类别数
batch_size = 100   # 数据批量
num_epochs = 2    # 数据批量
learning_rate = 0.003   # 学习率

# 数据加载
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# 定义双向的LSTM  many2one
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)   # 针对不同的设备初始化不同的张量
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 只提取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 实例化模型
model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)   # 默认加载的标签为one_hot

        outputs = model(images)
        loss = criterion(outputs, labels)   # 计算损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# 测试
correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, sequence_length, input_size).to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('测试集的准确率: {} %'.format(100 * correct / total))


# 保存模型
torch.save(model.state_dict(), 'model.ckpt')
