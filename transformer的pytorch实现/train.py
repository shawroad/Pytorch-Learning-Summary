"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-24
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataloader import MyDataSet
from config import set_args
from model import Transformer


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)



def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


if __name__ == '__main__':
    args = set_args()
    # S: 解码的开始, E: 解码的结束, P: padding
    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # 输入的词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    src_vocab_size = len(src_vocab)

    # 输出的词表
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}   # 解码时需要
    tgt_vocab_size = len(tgt_vocab)

    # 输入和输出的最大长度
    src_len = 5  # enc_input max sequence length
    tgt_len = 6  # dec_input(=dec_output) max sequence length

    # 将数据转为id序列
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

    loader = DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)

    model = Transformer()

    # 指定多gpu运行
    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.device_count() > 1:
        args.n_gpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(30):
        # 训练三十轮
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            if torch.cuda.is_available():
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存模型
        os.makedirs(args.save_model, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.save_model, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
