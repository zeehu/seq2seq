#coding: utf-8
#Created Time: 2020-09-10 10:54:37

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import config as cfg

from instructor import BasicInstructor
from model import Seq2Seq
from text_process import idx2word

class Seq2SeqInstructor(BasicInstructor):
    def __init__(self, opt):
        super(Seq2SeqInstructor, self).__init__(opt)
        self.seq2seq = Seq2Seq(opt.vocab_size, opt.embed_dim, 
                opt.num_hiddens, opt.num_layers, opt.drop_prob)
        if cfg.CUDA:
            self.seq2seq = self.seq2seq.cuda()
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=opt.lr)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def train(self, epoch, data_loader):
        l_sum = 0.0
        for i, data in enumerate(data_loader):
            #输入,输出为文本; pad至相同长度
            src, tgt = data['input'], data['target']

            #输出开始位置加<sos>, decoder解码需要
            #add sos_token inx at start
            s_tensor = torch.full([len(tgt), 1], self.w2i_dict[cfg.sos_token],
                    dtype=torch.long)
            tgt = torch.cat((s_tensor, tgt), dim=1)

            if epoch == 0 and i == 0:
                tmp = [src[0], tgt[0]]
                print(tmp)
                res = idx2word(tmp, self.i2w_dict)
                print("{}\t{}".format(" ".join(res[0]), " ".join(res[1])))

            if cfg.CUDA:
                src = src.cuda()
                tgt = tgt.cuda()

            self.optimizer.zero_grad()
            #src, tgt:[batch_size, seq_len]
            #outputs:[batch_size, seq_len, vocab_size]
            outputs = self.seq2seq(src, tgt)

            if cfg.CUDA:
                outputs = outputs.cuda()
            
            ignore_index = self.w2i_dict[cfg.pad_token]
            #loss = self.loss(outputs[1:].view(-1, cfg.vocab_size),
            loss = F.nll_loss(outputs.contiguous().view(-1, cfg.vocab_size),
                    tgt[:, 1:].contiguous().view(-1), #去掉<sos>
                    ignore_index=ignore_index)
            loss.backward()
            #梯度裁剪
            #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            self.optimizer.step()
            l_sum  += loss.data.item()
        if epoch % 1 == 0:
            print("epoch %d, loss %.4f" % (epoch + 1, l_sum / len(data_loader)))

    def _run(self):
        for epoch in range(self.opt.num_epochs):
            self.train(epoch, self.train_data.loader)
            #self.test()
