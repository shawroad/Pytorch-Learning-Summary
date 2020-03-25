"""

@file  : bert实现生成-OneByOne.py

@author: xiaolu

@time  : 2020-03-24

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
# from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from pytorch_pretrained_bert import BertAdam, BertForMaskedLM, BertTokenizer


bert_model_path = './pytorch_model.bin'
bert_config_path = './bert_config.json'
vocab_path = './vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_path)
# config = BertConfig.from_pretrained(bert_config_path)

model = BertForMaskedLM.from_pretrained('./Bert_Generation.tar.gz')

example_pair = dict()

input_text = "[CLS] I go to school by bus [SEP] "
target_text = "我搭公車上學"


for i in range(0, len(target_text) + 1):
    tokenized_text = tokenizer.tokenize(input_text)
    tokenized_text.extend(target_text[:i])
    tokenized_text.append('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])

    loss_ids = [-1] * (len(tokenizer.convert_tokens_to_ids(tokenized_text)) - 1)
    if i == len(target_text):
        loss_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[SEP]'))[0])
    else:
        loss_ids.append(tokenizer.convert_tokens_to_ids(target_text[i])[0])
    #   for _ in range(512-len(loss_ids)):
    #     loss_ids.append(-1)
    loss_tensors = torch.tensor([loss_ids])

    example_pair[tokens_tensor] = loss_tensors
    print(tokenized_text, loss_ids, loss_ids[-1])
    print(len(indexed_tokens), len(loss_tensors))

# Prepare optimizer
param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=300)


model.train()
for i in range(0, 3):
    eveloss = 0
    for k, v in example_pair.items():
        loss = model(k, masked_lm_labels=v)
        eveloss += loss.mean().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("step "+ str(i) + " : " + str(eveloss))

torch.save(model.state_dict(), './my_bert_gen.bin')

model.eval()
for k,v in example_pair.items():
    predictions = model(k)
    predicted_index = torch.argmax(predictions[0, len(predictions[0])-1]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    # if '[SEP]' in predicted_token:
    #   break
    print(predicted_token)
