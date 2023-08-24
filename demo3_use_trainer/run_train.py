"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-08-24
"""
import pandas as pd
from model import Model
from config import set_args
from data_helper import CustomDataset, Collator
from transformers.models.bert import BertTokenizer
from transformers import TrainingArguments, Trainer


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    correct_num = (predictions == labels).sum()
    correct_total = len(labels)
    return {"accuracy：": correct_num / correct_total}


if __name__ == '__main__':
    import os
    os.environ["WANDB_DISABLED"] = "true"

    args = set_args()
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    
    # 1. 加载数据
    df = pd.read_csv(args.train_data_path)
    df = df.head(10000).reset_index(drop=True)   # 用一万条数据做测试
    val_df = df.head(500).reset_index(drop=True)
    train_df = df.tail(df.shape[0] - 500).reset_index(drop=True)

    train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    evalute_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
        
    data_collator = Collator(tokenizer, args.max_seq_length)

    model = Model()

    train_args = TrainingArguments(output_dir="./checkpoints",  # 输出文件夹
                                   num_train_epochs=1,   # 训练轮数
                                   per_device_train_batch_size=64,  # 训练时的batch_size
                                   per_device_eval_batch_size=64,  # 验证时的batch_size
                                   logging_steps=10,  # log 打印的频率
                                   evaluation_strategy="epoch",  # 评估策略
                                   save_strategy="epoch",  # 保存策略  steps
                                   save_total_limit=3,  # 最大保存数
                                   learning_rate=2e-5,  # 学习率
                                   weight_decay=0.01,  # weight_decay
                                   load_best_model_at_end=True,
                                   gradient_accumulation_steps=1,
                                   seed=43)  # 训练完成后加载最优模型
                                                            

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=evalute_dataset,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics)

    trainer.train()
    
    trainer.evaluate()
                    

