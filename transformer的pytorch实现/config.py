"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-24
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", default=512, type=int, help='embedding dim')
    parser.add_argument("--d_ff", default=2048, type=int, help='FeedForward dimension')
    parser.add_argument("--d_k", default=64, type=int, help="K's dim")
    parser.add_argument("--d_v", default=64, type=int, help="V's dim")
    parser.add_argument("--n_layers", default=6, type=int, help="number of Encoder of Decoder Layer")
    parser.add_argument("--n_heads", default=8, type=int, help="number of heads in Multi-Head Attention")
    parser.add_argument("--src_vocab_size", default=6, type=int, help="")
    parser.add_argument("--tgt_vocab_size", default=9, type=int, help="")
    parser.add_argument("--save_model", default='save_model', type=str, help="")
    args = parser.parse_args()
    return args
