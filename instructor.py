# -*- coding: utf-8 -*-
# @Author       : Jezeehu
# @Project      : seq2seq
# @FileName     : instructor.py
# @Time         : Created at 2020-08-26

import numpy as np
import torch
import torch.nn as nn

import config as cfg
#from data_loader import DataIter
from helpers import create_logger
from text_process import load_dict, word2idx, idx2word
from data_loader import DataIter

class BasicInstructor:
    def __init__(self, opt):
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=cfg.log_filename) 
        
        self.w2i_dict, self.i2w_dict = load_dict()
        self.train_data = DataIter(opt.train_data, self.w2i_dict)
        cfg.vocab_size = len(self.w2i_dict)
        opt.vocab_size = cfg.vocab_size
        self.opt = opt
        self.show_config()

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        pass

    def show_config(self):
        self.log.info(100 * '=')
        self.log.info('> training arguments:')
        for arg in vars(self.opt):
            self.log.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        self.log.info(100 * '=')

