"""

@file  : bert+lstm实现生成.py

@author: xiaolu

@time  : 2020-03-24

"""
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch import nn


class BertLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # modelpath = "bert-base-chinese"
        self.bert = BertModel.from_pretrained('./Bert_Generation.tar.gz')
        self.rnn = nn.LSTM(num_layers=2, dropout=0.2, input_size=768, hidden_size=768 // 2)
        self.fc = nn.Linear(384, self.bert.config.vocab_size)

    def forward(self, x, y=None):
        self.bert.train()
        encoded_layers, _ = self.bert(x)
        print(encoded_layers)
        for i in encoded_layers:
            enc, _ = self.rnn(i)
        logits = self.fc(enc)
        if y is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), y.view(-1))
            return loss
        return logits


if __name__ == '__main__':
    input_text = '[CLS] I go to school by bus [SEP]'
    target_text = '我搭公車上學'

    tokenizer = BertTokenizer.from_pretrained('./vocab.txt')
    example_pair = dict()

    # 数据预处理
    for i in range(0, len(target_text) + 1):
        tokenized_text = tokenizer.tokenize(input_text)  # 对输入文本分词
        tokenized_text.extend(target_text[:i])   # 每次的输入加一步解码的信息
        tokenized_text.append('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # 用-1标记的都是不求损失的　
        loss_ids = [-1] * (len(tokenizer.convert_tokens_to_ids(tokenized_text)) - 1)
        if i == len(target_text):
            loss_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[SEP]'))[0])  # 最后加个sep
        else:
            loss_ids.append(tokenizer.convert_tokens_to_ids(target_text[i])[0])  # 这里加入的是当前需要预测的标签

        loss_tensors = torch.tensor([loss_ids])

        example_pair[tokens_tensor] = loss_tensors

    model = BertLSTM()
    optimizer = torch.optim.Adamax(model.parameters(), lr=5e-5)

    model.train()
    for i in range(0, 150):
        eveloss = 0
        for k, v in example_pair.items():
            optimizer.zero_grad()
            loss = model(k, v)
            exit()

    model.train()
    for i in range(0, 150):
        eveloss = 0
        for k, v in example_pair.items():
            optimizer.zero_grad()
            loss = model(k, v)
            eveloss += loss.mean().item()
            loss.backward()
            optimizer.step()
        print("step " + str(i) + " : " + str(eveloss))

    model.eval()
    for k, v in example_pair.items():
        predictions = model(k)
        predicted_index = torch.argmax(predictions[0, -1]).item()
        if predicted_index < model.bert.config.vocab_size:
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        print(predicted_token)

