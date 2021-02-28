"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-28
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", default=768, type=int, help='embedding dim')
    parser.add_argument("--maxlen", default=30, type=int, help="maxlen")
    parser.add_argument("--n_segments", default=2, type=int, help="")
    parser.add_argument("--d_k", default=64, type=int, help="Q and K's dim")
    parser.add_argument("--d_v", default=64, type=int, help="V's dim")
    parser.add_argument("--n_heads", default=12, type=int, help="number of heads in Multi-Head Attention")
    parser.add_argument("--d_ff", default=768*4, type=int, help="FeedForward dimension")
    parser.add_argument("--n_layers", default=6, type=int, help="number of Encoder of Decoder Layer")
    parser.add_argument("--batch_size", default=6, type=int, help="")
    parser.add_argument("--vocab_size", default=29, type=int, help='embedding dim')
    parser.add_argument("--save_model", default='save_model', type=str, help="")
    parser.add_argument("--max_pred", default=15, type=int, help='mask多少词')
    args = parser.parse_args()
    return args
