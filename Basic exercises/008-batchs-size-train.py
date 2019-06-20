"""

@file   : 008-batchs-size-train.py

@author : xiaolu

@time   : 2019-06-20

"""
import torch
import torch.utils.data as Data

torch.manual_seed(43)

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# print(x)    # tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
# print(y)    # tensor([10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.])

torch_dataset = Data.TensorDataset(x, y)   # 将特征和标签组成一个元组
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,      # 加载数据需要几个子进程
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # 训练数据
            print("Epoch:{}, step:{}, batch_x:{}, batch_y:{}".format(epoch, step, batch_x.numpy(), batch_y.numpy()))


if __name__ == '__main__':
    show_batch()

