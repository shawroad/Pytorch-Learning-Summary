"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-08-24
"""
import os
import time
import pandas as pd
import deepspeed
import torch
import random
from tqdm import tqdm
import numpy as np
from torch import nn
from config import set_args
from model import Model
from sklearn import metrics
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from data_helper import CustomDataset, Collator, clean_text
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def evaluate():
    eval_targets = []
    eval_predict = []
    model.eval()
    for step, batch in tqdm(enumerate(val_dataloader)):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():   # 不进行梯度计算
            logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
    eval_accuracy = metrics.accuracy_score(eval_targets, eval_predict)
    return eval_accuracy



deepspeed_config = {
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5
        }
    },
    "fp16": {
        "enabled": False  # True
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True
    },
    "wall_clock_breakdown": True,
    "log_dist": False,
}



if __name__ == '__main__':
    args = set_args()
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 1. 加载数据集
    df = pd.read_csv(args.train_data_path)
    df = df.head(10000).reset_index(drop=True)   # 用一万条数据做测试

    val_df = df.head(500).reset_index(drop=True)
    train_df = df.tail(df.shape[0] - 500).reset_index(drop=True)
    # print(train_df.shape)   # (59500, 3)
    # print(val_df.shape)   # (500, 3)

    collate_fn = Collator()

    # 训练数据集准备
    train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)

    # 验证集准备
    val_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=args.val_batch_size, collate_fn=collate_fn)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    model_engine, optimzier, _, _ = deepspeed.initialize(config=deepspeed_config, model=model, model_parameters=model.parameters())
    loss_func = nn.CrossEntropyLoss()

    global_step = 0
    epoch_loss = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        s_time = time.time()
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # print(input_ids.size())   # torch.Size([8, 80])
            # print(input_mask.size())   # torch.Size([8, 80])
            # print(segment_ids.size())   # torch.Size([8, 80])
            # print(label_ids.size())   # torch.Size([8])
            logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            loss = loss_func(logits, label_ids)
    
            model_engine.backward(loss)
            # loss.backward()
            print("epoch:{}, step:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            epoch_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                model_engine.step()
                
                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()

            train_label.extend(label_ids.cpu().detach().numpy().tolist())
            train_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
        e_time = time.time()
        print('平均每个batch的耗时:', (e_time - s_time) / global_step)

        train_accuracy = metrics.accuracy_score(train_label, train_predict)
        eval_accuracy = evaluate()

        s = 'Epoch: {} | Loss: {:10f} | Train acc: {:10f} | Val acc: {:10f} '
        ss = s.format(epoch, epoch_loss / global_step, train_accuracy, eval_accuracy)

        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            ss += '\n'
            f.write(ss)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
