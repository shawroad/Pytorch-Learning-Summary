"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-08-24
"""
import re
import torch
import json
import pandas as pd
from torch.utils.data import Dataset


'''
def clean_text(text):
    rule_url = re.compile(
        '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    )
    rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5]')
    rule_space = re.compile('\\s+')
    text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
    text = rule_url.sub(' ', text)
    text = rule_legal.sub(' ', text)
    text = rule_space.sub(' ', text)
    return text.strip()
'''


def clean_text(text):
    # 去除url
    rule_url = re.compile(
        '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    )
    # 去除杂乱的名字
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)   # 去重中间杂乱的名字

    rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5，！”？?~《》。#、；：“（）]')

    rule_space = re.compile('\\s+')
    text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
    text = rule_url.sub(' ', text)
    text = rule_legal.sub(' ', text)
    text = rule_space.sub(' ', text)
    return text.strip()


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.label = dataframe.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.text[index],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }





class Collator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # 按batch进行padding获取当前batch中最大长度
        # max_len = max([len(d['input_ids']) for d in batch])
        # if max_len > 512:
        #     max_len = 512

        # 定一个全局的max_len
        # max_len = 128
        max_len = 256

        input_ids, attention_mask, token_type_ids, labels = [], [], [], []
        for item in batch:
            input_ids.append(self.pad_to_maxlen(item['input_ids'], max_len=max_len, pad_value=self.pad_token_id))
            attention_mask.append(self.pad_to_maxlen(item['attention_mask'], max_len=max_len, pad_value=self.pad_token_id))
            token_type_ids.append(self.pad_to_maxlen(item['token_type_ids'], max_len=max_len, pad_value=self.pad_token_id))
            labels.append(item['label'])

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
        all_label_ids = torch.tensor(labels, dtype=torch.long)
        inputs = {
            'input_ids': all_input_ids,
            'attention_mask': all_input_mask,
            'token_type_ids': all_segment_ids,
            'labels': all_label_ids
        }
        return inputs

    def pad_to_maxlen(self, input_ids, max_len, pad_value=0):
        if len(input_ids) >= max_len:
            input_ids = input_ids[:max_len]
        else:
            input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
        return input_ids