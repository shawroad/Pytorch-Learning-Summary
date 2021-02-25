"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-24
"""
import torch
from config import set_args
from torch.utils.data.dataloader import DataLoader
from dataloader import MyDataSet
from model import Transformer


def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # print(enc_outputs.size())   # torch.Size([1, 5, 512])  编码的输出
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


if __name__ == '__main__':
    args = set_args()

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

    # 加载模型
    model = Transformer()
    model.load_state_dict(torch.load('./save_model/epoch4_ckpt.bin', map_location='cpu'))
    print("模型加载成功...")
    model.eval()

    # Test
    enc_inputs, _, _ = next(iter(loader))
    greedy_dec_input = greedy_decoder(model, enc_inputs[0].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[0].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[0], '->', [idx2word[n.item()] for n in predict.squeeze()])
