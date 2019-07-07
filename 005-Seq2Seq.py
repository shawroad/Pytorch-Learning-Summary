"""

@file   : 005-Seq2Seq.py

@author : xiaolu

@time   : 2019-07-07

"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

dtype = torch.FloatTensor

# S: 解码的开始标志
# E: 解码结束标志
# P: 填充  如果序列没有达到要求的长度 进行填充  也就padding

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']

char2id = {char: i for i, char in enumerate(char_arr)}


# [[输入序列， 输出序列]， [输入序列, 输出序列], ''''']
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]


n_step = 5    # 序列都需要填充成这么长
n_hidden = 128    # 隐层输出维度
n_class = len(char2id)    # 包含的字符  为了解码需要
batch_size = len(seq_data)


def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []
    for seq in seq_data:
        for i in range(2):
            # 这里将输入序列和输出序列都填充为长度为5的序列
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))   # 长度不够n_step进行填充P

        # 将序列转为id序列
        input = [char2id[n] for n in seq[0]]    # 这里是编码的输入
        output = [char2id[n] for n in ('S' + seq[1])]     # 这里是解码的每步输入
        target = [char2id[n] for n in (seq[1] + 'E')]    # 这里是解码每步的输出

        input_batch.append(np.eye(n_class)[input])   # 将输入转为one_hot
        output_batch.append(np.eye(n_class)[output])  # 将输出转为one_hot
        target_batch.append(target)    # torch中算交叉损失熵的时候标签不需要转one_hot

    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))


# 定义模型
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        self.encoder_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.decoder_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, encoder_input, encoder_hidden, decoder_input):
        encoder_input = encoder_input.transpose(0, 1)    # [max_len(=n_step, time step), batch_size, n_class]
        decoder_input = decoder_input.transpose(0, 1)    # [max_len(=n_step, time step), batch_size, n_class]

        _, encoder_states = self.encoder_cell(encoder_input, encoder_hidden)
        outputs, _ = self.decoder_cell(decoder_input, encoder_states)   # 编码结果和解码的每一步输入联合起来去得到每步的解码的结果

        model = self.fc(outputs)
        return model


# 获取批量数据
input_batch, output_batch, target_batch = make_batch(seq_data)

model = Seq2Seq()
criterion = nn.CrossEntropyLoss()   # 定义损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(5000):
    # 初始化一个编码的隐层的向量
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))

    optimizer.zero_grad()
    output = model(input_batch, hidden, output_batch)
    output = output.transpose(0, 1)

    loss = 0
    for i in range(0, len(target_batch)):
        loss += criterion(output[i], target_batch[i])

    if (epoch + 1) % 100 == 0:
        print("Epoch:{} cost={}".format(epoch+1, loss))

    loss.backward()
    optimizer.step()


# 测试
def translate(word):
    input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])   # 输出给个空值 即P*len(word)

    hidden = Variable(torch.zeros(1, 1, n_hidden))
    output = model(input_batch, hidden, output_batch)

    predict = output.data.max(2, keepdim=True)[1]    # 选概率最大的索引
    decoded = [char_arr[i] for i in predict]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated.replace('P', '')


print('test')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('upp ->', translate('upp'))
