"""
@file   : dataloader.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-24
"""
from torch.utils.data.dataset import Dataset


class MyDataSet(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

