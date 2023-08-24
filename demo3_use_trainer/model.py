"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-08-24
"""
import torch
from torch import nn
from config import set_args
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertModel

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
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True
        )

        output1 = bert_output[1]
        logits = self.classifier(output1)

        loss = self.loss_fct(logits.view(-1, 2), labels.view(-1))
        return {'loss': loss, 'logits': logits}

        # if labels:
        #     loss = self.loss_fct(logits.view(-1, 2), labels.view(-1))
        #     return {'loss':loss, 'logits': logits}
        # else:
        #     return logits