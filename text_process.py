# -*- coding: utf-8 -*-
# @Author       : Jezeehu
# @Project      : seq2seq
# @FileName     : text_process.py
# @Time         : Created at 2020-08-26
import sys
 
import numpy as np
import os
import torch

import config as cfg

def get_pair_list(file):
    in_list = list()
    out_list = list()
    with open(file) as raw:
        for line in raw:
            li_sp = line.strip().split('\t')
            in_list.append(li_sp[0].lower().split(' '))
            out_list.append(li_sp[1].lower().split(' '))
    return in_list, out_list

def get_word_list(file):
    """get word set"""
    word_set = set()
    with open(file) as raw:
        for line in raw:
            li_sp = line.strip().split("\t")
            for sentence in li_sp:
                for word in sentence.lower().split(' '):
                    word_set.add(word)
    return list(word_set)


def get_dict(word_set):
    """get word2idx_dict and idx2word_dict"""
    word2idx_dict ={cfg.pad_token : 0,
                    cfg.sos_token : 1,
                    cfg.eos_token : 2,
                    cfg.unk_token : 3}
    idx2word_dict = {0 : cfg.pad_token,
                     1 : cfg.sos_token,
                     2 : cfg.eos_token,
                     3 : cfg.unk_token}
    index = len(word2idx_dict)
    for word in word_set:
        word2idx_dict[word] = index
        idx2word_dict[index] = word
        index += 1
    return word2idx_dict, idx2word_dict

# ============================================
def init_dict():
    """
    Initialize dictionaries of dataset, please note that '0': padding_idx, '1': start_letter.
    Finally save dictionary files locally.
    """
    word_set = get_word_list(cfg.train_data)
    word2idx_dict, idx2word_dict = get_dict(word_set)

    with open('{}_wi_dict.txt'.format(cfg.train_data), 'w') as dictout:
        dictout.write(str(word2idx_dict))
    with open('{}_iw_dict.txt'.format(cfg.train_data), 'w') as dictout:
        dictout.write(str(idx2word_dict))

    print('total tokens: ', len(word2idx_dict))


def load_dict():
    """Load dictionary from local files"""
    iw_path = '{}_iw_dict.txt'.format(cfg.train_data)
    wi_path = '{}_wi_dict.txt'.format(cfg.train_data)

    if not os.path.exists(iw_path) or not os.path.exists(iw_path):  # initialize dictionaries
        init_dict()

    with open(iw_path, 'r') as dictin:
        idx2word_dict = eval(dictin.read().strip())
    with open(wi_path, 'r') as dictin:
        word2idx_dict = eval(dictin.read().strip())

    return word2idx_dict, idx2word_dict


def load_test_dict(dataset):
    """Build test data dictionary, extend from train data. For the classifier."""
    word2idx_dict, idx2word_dict = load_dict()  # train dict
    word_set = get_word_list(cfg.test_data)
    index = len(word2idx_dict)  # current index

    # extend dict with test data
    for word in word_set:
        if word not in word2idx_dict:
            word2idx_dict[word] = str(index)
            idx2word_dict[str(index)] = word
            index += 1
    return word2idx_dict, idx2word_dict


def idx2word(tensor, dictionary):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            if word == 0:
                break
            sent_token.append(dictionary[word])
        tokens.append(sent_token)
    return tokens


def word2idx(tokens, dictionary, add_eos=False):
    """transform word tokens to Tensor"""
    global i
    tensor = []
    print(len(tokens))
    n = 0
    for sent in tokens:
        n += 1
        t = 0
        sent_ten = []
        for i, word in enumerate(sent):
            t += 1
            if word == cfg.pad_token:
                break
            try:
                sent_ten.append(dictionary[str(word)])
            except:
                print(sent)
                print(n)
                print(t)
                continue
        if i >= cfg.max_seq_len - 1:
            #超长sentence 丢弃
            continue
        if add_eos:
            sent_ten.append(dictionary[cfg.eos_token])
        for _ in range(cfg.max_seq_len - len(sent_ten)):
            sent_ten.append(dictionary[cfg.pad_token])
        tensor.append(sent_ten)
    return torch.LongTensor(tensor)


def padding(tokens):
    """pad sentences with padding_token"""
    max_len = max([len(s) for s in tokens])
    pad_idx = 0
    pad_tokens = [sent + [pad_idx] * (ml - len(sent)) for sent in tokens]
    return pad_tokens

if __name__ == '__main__':
    os.chdir('../')
    # process_cat_text()
    # load_test_dict('mr15')
    # extend_clas_train_data()
    pass
