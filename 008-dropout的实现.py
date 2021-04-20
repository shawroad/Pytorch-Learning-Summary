"""
@file   : 008-dropout的实现.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-20
"""
import torch


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)  # 全部丢弃
    if dropout == 0:
        return X    # 全部不丢弃
    mask = (torch.randn(X.shape) > dropout).float()  # 0-1之间的均匀分布
    # 如果dropout=0.3  则丢弃70%的输出
    return mask * X / (1.0 - dropout)


if __name__ == '__main__':
    X = torch.randn(16, dtype=torch.float32).reshape(2, 8)
    print(X)
    print(dropout_layer(X, 0.3))
    print(dropout_layer(X, 0))

