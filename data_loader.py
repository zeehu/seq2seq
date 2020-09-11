# -*- coding: utf-8 -*-
# @Author       : Jezeehu
# @Project      : seq2seq
# @FileName     : data_loader.py
# @Time         : Created at 2020-08-26

import random
from torch.utils.data import DataLoader

from text_process import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DataIter:
    def __init__(self, samples, w2i_dict, shuffle=None):
        self.w2i_dict = w2i_dict
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=Dataset(self.__read_data__(samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

    def __read_data__(self, samples):
        """
        input, output: end with end_letter.
        """
        inp, out = get_pair_list(samples)
        inp = word2idx(inp, self.w2i_dict)
        out = word2idx(out, self.w2i_dict)

        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, out)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]
