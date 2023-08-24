"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-08-24
"""
import torch
from torch import nn
from config import set_args
from transformers.models.bert import BertConfig
from transformers import BertTokenizer, ErnieModel, BertModel
args = set_args()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dense1 = torch.nn.Linear(768, 384)
        self.dense2 = torch.nn.Linear(384, 128)
        self.dense3 = torch.nn.Linear(128, 2)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrained_model_path)
        self.classifier = Classifier()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True
        )
        # print(bert_output[0].size())   # torch.Size([8, 80, 768])
        # print(bert_output[1].size())  # torch.Size([8, 768])

        # # 每个多头注意力矩阵的输出
        # len(bert_output[3]) = 12
        # bert_output[3][0].size() = torch.Size([16, 12, 256, 256])
        output1 = bert_output[1]
        logits = self.classifier(output1)
        # return logits, bert_output[3], bert_output[2]
        # logits: batch_size, 2
        # print(logits.size())   # torch.Size([8, 2])
        return logits