#coding: utf-8
#Created Time: 2020-09-10 11:04:03
import os
import re
import time
import torch
from time import strftime, localtime

# ===Program===
CUDA = True
data_shuffle = True

train_data = 'data/train_data'
test_data = 'data/test_data'
vocab_size = 5000
max_seq_len = 20
num_epochs = 10
batch_size = 2
lr = 0.01
embed_dim = 100
num_hiddens = 128
num_layers = 3
drop_prob = 0.0

pad_token = '<pad>'
sos_token = '<sos>'
eos_token = '<eos>'
unk_token = '<unk>'

# ===log===
log_time_str = strftime("%m%d_%H%M_%S", localtime())
log_filename = strftime("log/log_%s" % log_time_str)
if os.path.exists(log_filename + '.txt'):
    i = 2
    while True:
        if not os.path.exists(log_filename + '_%d' % i + '.txt'):
            log_filename = log_filename + '_%d' % i
            break
        i += 1
log_filename = log_filename + '.txt'

# Automatically choose GPU or CPU
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    os.system('nvidia-smi -q -d Utilization > gpu')
    with open('gpu', 'r') as _tmpfile:
        util_gpu = list(map(int, re.findall(r'Gpu\s+:\s*(\d+)\s*%', _tmpfile.read())))
    os.remove('gpu')
    CUDA = True
    if len(util_gpu):
        device = util_gpu.index(min(util_gpu))
    else:
        device = 0
else:
    CUDA = False
    device = -1
# device=1
# print('device: ', device)
torch.cuda.set_device(device)

# ===Save Model and samples===
save_root = 'save/{}/'.format(time.strftime("%Y%m%d"))

# Init settings according to parser
def init_param(opt):
    global data_shuffle, vocab_size, num_epochs, batch_size, \
    max_seq_len, lr, train_data, test_data, embed_dim, num_hiddens, \
    num_layers, drop_prob, attention_size, save_root

    data_shuffle = opt.shuffle

    vocab_size = opt.vocab_size
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    max_seq_len = opt.max_seq_len
    lr = opt.lr
    embed_dim = opt.embed_dim
    num_hiddens = opt.num_hiddens
    num_layers = opt.num_layers
    drop_prob = opt.drop_prob

    train_data = opt.train_data
    test_data = opt.test_data

    # Create Directory
    dir_list = ['save', 'log']
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
