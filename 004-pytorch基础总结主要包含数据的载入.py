"""

@file   : 004-pytorch基础总结主要包含数据的载入

@author : xiaolu

@time   : 2019-06-21

"""
import torch
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

# 对标量求导
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)
y = w * x + b
y.backward()    # 这里默认y.backward(torch.Tensor([1]))
print(x.grad)
print(w.grad)
print(b.grad)
print("+"*100)


# 对矩阵求导
x = torch.randn(3)  # shape=(1, 3)的随机数  范围为-1， 1
x = Variable(x, requires_grad=True)
# print(x.data.numpy())
y = x * 2
y.backward(torch.Tensor([1, 0.1, 0.01]))   # 各个方向上梯度成这个里面的数
print(x.grad)
print("+"*100)


# 通过ImageFolder载入我们的数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)
# mnist目录下是文件夹('1', '2', '3'...) 每个文件夹下放的是图片
dataset = ImageFolder(root='./mnist/', transform=transform)
# ImageFolder参数介绍
# root：在root指定的路径下寻找图片
# transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
# target_transform：对label的转换
# loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象

print(dataset.classes)   # 根据分的文件夹的名字来确定的类别
print(dataset.class_to_idx)  # 按顺序为这些类别定义索引为0,1...
print(dataset.imgs)      # 返回从所有文件夹中得到的图片的路径以及其类别

print(dataset[0])       # dataset中数据方的格式(数据, 标签)
print(dataset[0][0])    # 打印的数据
print(dataset[0][1])    # 打印的是标签
print(dataset[0][0].size())
print("+"*100)


# 通过Dataset加载我们的文本数据集
class MyDataset(Dataset):
    def __init__(self, file_name):
        with open(file_name, 'r') as f:
            data_temp = f.readlines()
            data_temp = [data.replace('\n', '') for data in data_temp]
        self.txt_data = data_temp

    def __len__(self):
        return len(self.txt_data)

    def __getitem__(self, idx):
        return self.txt_data[idx]


# 然后制造批数据
dataset = DataLoader(MyDataset('./cmn.txt'), batch_size=3)
print(dataset.batch_size)   # 设置的批次
for step, d in enumerate(dataset):
    print("step:", step)
    print("Batch_data:", d)


