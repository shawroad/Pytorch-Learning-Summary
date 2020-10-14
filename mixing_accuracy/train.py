# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 9:33
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm
import os
import gzip
import torch
import random
import pickle
import time
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from model import Model
from config import set_args
from apex import amp


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate(epoch):
    print("***** Running evaluating *****")
    print("  Num examples = {}".format(len(eval_features)))
    print("  Batch size = {}".format(args.eval_batch_size))

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()
        with torch.no_grad():
            loss, logits = model(input_ids, input_mask, segment_ids, labels=label_ids)

        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去

        labels = label_ids.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    eval_recall = recall_score(labels_all, predict_all)
    eval_precision = precision_score(labels_all, predict_all)
    s = 'epoch:{}, eval_loss: {}, eval_precision: {}, eval_accuracy:{}, eval_recall:{}'.format(epoch, eval_loss, eval_precision, eval_accuracy, eval_recall)
    print(s)
    s += '\n'
    with open('result_eval.txt', 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


if __name__ == '__main__':
    args = set_args()
    set_seed(args)  # 设定随机种子

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # 加载训练集
    with gzip.open(args.train_features_path, 'rb') as f:
        train_features = pickle.load(f)

    with gzip.open(args.eval_features_path, 'rb') as f:
        eval_features = pickle.load(f)

        # Prepare Optimizer
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = Model()
    
    # 指定多gpu运行
    if torch.cuda.is_available():
        model.cuda()

    # if torch.cuda.device_count() > 1:
    #     args.n_gpu = torch.cuda.device_count()
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # 就这一行
    #     model = nn.DataParallel(model)

    # 模型
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    

    if torch.cuda.device_count() > 1:
        args.n_gpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)


    best_loss = None
    global_step = 0

    if args.do_train:
        print("***** Running training *****")
        print("  Num examples = {}".format(len(train_features)))
        print("  Batch size = {}".format(args.train_batch_size))
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        model.train()

        for epoch in range(args.num_train_epochs):
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, labels_ids = batch
                loss, logits = model(input_ids, input_mask, segment_ids, labels=labels_ids)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('epoch:{}, step:{}, loss:{:10f}, time_cost:{:10f}'.format(epoch, step, loss, time.time()-start_time))
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # 一个epoch跑完 然后进行验证
            os.makedirs(args.output_dir, exist_ok=True)
            output_prediction_file = os.path.join(args.output_dir, "epoch{}_prediction.csv".format(epoch))

            # 验证验证集
            test_loss, test_acc = evaluate(epoch)
            # 验证训练集中四万之后的数据
            model.train()

            if best_loss is None or best_loss > test_loss:
                best_loss = test_loss
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                os.makedirs(args.ckpt_dir, exist_ok=True)
                output_model_file = os.path.join(args.ckpt_dir, "best_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.ckpt_dir, "epoch{}_ckpt.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

