#coding: utf-8
#Created Time: 2020-09-09 15:55:13

import torch
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
    def __init__(self, embedding, embed_dim, num_hiddens, num_layers, \
            drop_prob=0.0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = embedding #encoder decoder共享embedding
        self.rnn = nn.GRU(embed_dim, num_hiddens, num_layers, \
                dropout=drop_prob)

    def forward(self, inputs, state):
        #inputs [batch_size, seq_len]
        #state  [None]

        #[batch_size, seq_len]=>[b_s, s_l, embed_size]=>[s_l, b_s, e_s]
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)

        #out:[s_l, b_s, ]
        #ht:[num_layers * num_directions, b_s, hidden_state]
        out, ht = self.rnn(embedding, state)
        return out, ht

    def begin_state(self):
        #初始hidden_state 为空
        return None

def Attention(nn.Module):
    def __init__(input_size, method='dot'):
        super(Attention, self).__init__()
        self.method = method
        self.input_size = input_size
        if self.method == 'general':
            self.atten = nn.Linear(self.input_size, self.input_size, bias=False)
        if self.method == 'concat':
            self.atten = nn.Sequential(
                    nn.Linear(self.input_size * 2, self.input_size, bias=False)
                    nn.Tanh()
                    nn.Linear(self.input_size, 1, bias=False))
            #多种实现方法
            #self.v = nn.Parameter(torch.FloatTensor(1, input_size))
            self.v = nn.Parameter(torch.rand(input_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

    def score(self, enc_states, dec_states):
        if self.method == 'dot':
            s = torch.dot(enc_states, dec_states)
            return s
        if self.method == 'general':
            s = self.atten(enc_states)
            s = torch.dot(dec_states, s)
            return s
        if self.method == 'concat':
            enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
            s = self.v * self.atten(enc_and_dec_states)
            return s

    def forward(self, enc_states, dec_states):
        #decoder隐藏状态广播到和编码器隐藏状态相同
        dec_states = dec_states.unsqueeze(dim=0).expand_as(enc_states)
        score = self.score(enc_states, dec_states)
        alpha = F.softmax(score, dim=0)
        c = (alpha * enc_states).sum(dim=0)
        return c

class Decoder(nn.Module):
    def __init__(self, embedding, vocab_size, embed_dim, num_hiddens,
            num_layers, attention_size, drop_prob=0.0):
        super(Decoder, self).__init__()
        self.embedding = embedding #共享embedding
        self.attention = Attention(num_hiddens)
        self.rnn = nn.GRU(num_hiddens + embed_dim, num_hiddens,
                num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        c = attention(enc_states, state[-1])
        inp_and_c = torch.cat((self.embedding(cur_input), c), dim=1)
        output, state = self.rnn(inp_and_c.unsqueeze(0), state)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_states):
        return enc_states
