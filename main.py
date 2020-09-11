# -*- coding: utf-8 -*-
# @Author       : Jezeehu
# @Project      : seq2seq
# @FileName     : main.py
# @Time         : Created at 2020-08-26
from __future__ import print_function

import argparse

import config as cfg

#from utils.text_process import load_test_dict, text_process

from seq2seq import Seq2SeqInstructor

def program_config(parser):
    # Program
    parser.add_argument('--shuffle', default=cfg.data_shuffle, type=int)
    parser.add_argument('--vocab_size', default=cfg.vocab_size, type=int)
    parser.add_argument('--num_epochs', default=cfg.num_epochs, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--max_seq_len', default=cfg.max_seq_len, type=int)
    parser.add_argument('--lr', default=cfg.lr, type=float)
    parser.add_argument('--train_data', default=cfg.train_data, type=str)
    parser.add_argument('--test_data', default=cfg.test_data, type=str)
    parser.add_argument('--embed_dim', default=cfg.embed_dim, type=int)
    parser.add_argument('--num_hiddens', default=cfg.num_hiddens, type=int)
    parser.add_argument('--num_layers', default=cfg.num_layers, type=int)
    parser.add_argument('--drop_prob', default=cfg.drop_prob, type=float)

    return parser


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opt = parser.parse_args()

    cfg.init_param(opt)

    inst = Seq2SeqInstructor(opt)
    inst._run()
