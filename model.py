#coding: utf-8
#Created Time: 2020-09-09 15:55:13

import torch
import torch.nn.functional as F
from torch import nn

import config as cfg

import math
import random
from queue import PriorityQueue
import operator

class Encoder(nn.Module):
    def __init__(self, embedding, embed_dim, num_hiddens, num_layers, \
            drop_prob=0.0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.embedding = embedding #encoder decoder共享embedding
        self.rnn = nn.GRU(embed_dim, num_hiddens, num_layers, \
                dropout=drop_prob)

    def forward(self, inputs, state=None):
        #inputs [batch_size, seq_len]
        #state  [None]
        #[batch_size, seq_len]=>[b_s, s_l, embed_dim]=>[s_l, b_s, e_d]
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        #embedding:[s_l, b_s, e_d]
        #state:None
        #out:[s_l, b_s, num_hidden]
        #ht:[num_layers * num_directions, b_s, n_h]
        out, ht = self.rnn(embedding, state)
        return out, ht

    def begin_state(self):
        #初始hidden_state 为空
        return None

class Attention(nn.Module):
    def __init__(self, input_size, method='dot'):
        super(Attention, self).__init__()
        self.method = method
        self.input_size = input_size
        if self.method == 'general':
            self.atten = nn.Linear(self.input_size, self.input_size, bias=False)
        if self.method == 'concat':
            self.atten = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size, bias=False),
                nn.Tanh())
            #多种实现方法
            #self.v = nn.Parameter(torch.FloatTensor(1, input_size))
            self.v = nn.Parameter(torch.rand(input_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

    def score(self, dec_states, enc_states):
        #pytorch rnn输出调换了b_s和s_l维,这里恢复
        #enc_states:[s_l, b_s, n_h]=>[b_s, s_l, n_h]
        enc_states = enc_states.permute(1, 0 , 2)
        if self.method == 'dot':
            #当前时刻decoder隐藏状态 * encoder隐藏状态
            #dec_states:[batch_size, num_hiddens]=>[b_s, 1, n_h]
            dec_states = dec_states.unsqueeze(1)
            #enc_states:[b_s, s_l, n_h]=>[b_s, n_h, s_l]
            enc_states = enc_states.permute(0, 2, 1)
            #s:[b_s, 1, s_l]
            s = torch.bmm(dec_states, enc_states)
            return s
        if self.method == 'general':
            #h_d * w * h_e
            s = self.atten(enc_states)
            #s:[b_s, s_l, n_h]=>[b_s, n_h, s_l]
            s = s.permute(0, 2, 1)
            dec_states = dec_states.unsqueeze(1)
            s = torch.bmm(dec_states, s)
            return s
        if self.method == 'concat':
            #decoder隐藏状态广播到和编码器隐藏状态相同
            dec_states = dec_states.unsqueeze(1).expand_as(enc_states)
            enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
            s = self.atten(enc_and_dec_states)
            s = s.permute(0, 2, 1)
            v = self.v.repeat(enc_states.size(0), 1).unsqueeze(1)
            s = torch.bmm(v, s)
            return s

    def forward(self, enc_states, dec_states):
        ##print("###### attention")
        ##print("enc_states:{}".format(enc_states.size()))
        ##print("dec_states:{}".format(dec_states.size()))
        #dec_states:[batch_size, num_hiddens]=>[seq_len, b_s, n_h]
        #print("broadcast dec_states:{}".format(dec_states.size()))
        #print("transpose dec_states:{}".format(dec_states.size()))
        score = self.score(dec_states, enc_states)
        ##print("score:{}".format(score.size()))
        ##print(score)
        #alpha:[b_s, 1, s_l]
        alpha = F.softmax(score, dim=2)
        ##print("alpha:{}".format(alpha.size()))
        ##print(alpha)
        #c:[b_s, 1, n_h]
        c = torch.bmm(alpha, enc_states.permute(1, 0, 2))
        ##print("c:{}".format(c.size()))
        ##print(c)
        return c

class Decoder(nn.Module):
    def __init__(self, embedding, vocab_size, embed_dim,
            num_hiddens,num_layers, drop_prob=0.0):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.embedding = embedding #共享embedding
        self.attention = Attention(num_hiddens, method='dot')
        self.rnn = nn.GRU(num_hiddens + embed_dim, num_hiddens,
                num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        #print("###### decoder")
        #print("input:{}".format(cur_input.size()))
        #print("state:{}".format(state.size()))
        #print("enc_states:{}".format(enc_states.size()))
        
        #cur_input:[batch_size, 1]
        #state:[num_layers * num_directions, b_s, n_h]
        #enc_states:[s_l, b_s, num_hidden]

        #c:[batch_size, 1, num_hidden]
        c = self.attention(enc_states, state[-1]) #取decoder最后一层隐藏状态做attention
        #print("c:{}".format(c.size()))
        #inp:[batch_size, 1, embed_dim]
        #print("cur_input:{}".format(cur_input.size()))
        
        #inp:[batch_size, 1, vocab_size]
        inp = self.embedding(cur_input)
        #print("inp:{}".format(inp.size()))

        #inp_and_c:[batch_size, 1, embed_dim+num_hidden]
        #输入rnn做转置 [1, b_s, e_d+n_h]
        inp_and_c = torch.cat((inp, c), dim=2).transpose(0, 1)
        #print("inp_and_c:{}".format(inp_and_c.size()))
        output, state = self.rnn(inp_and_c.contiguous(), state.contiguous())
        #print("output:{}".format(output.size()))
        #print("state:{}".format(state.size()))

        #output:[batch_size, 1, vocab_size]
        output = self.out(output) #.squeeze(dim=0)
        #print("output:{}".format(output.size()))
        return output, state

    def begin_state(self, enc_states):
        return enc_states

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hiddens, num_layers,
            drop_prob=0.0):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(self.embedding, embed_dim, num_hiddens,
                num_layers, drop_prob)
        self.decoder = Decoder(self.embedding, vocab_size, embed_dim,
                num_hiddens, num_layers, drop_prob)

    def forward(self, src, tgt, teacher_forcing_ratio=1):
        batch_size = src.size(0)
        max_len = tgt.size(1) - 1 #去除<sos>
        vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(max_len, batch_size, vocab_size)
        #src:[b_s, s_l]
        #enc_outputs:[s_l, b_s, num_hidden]
        #enc_states:[num_layers * num_directions, b_s, n_h]
        enc_outputs, enc_states = self.encoder(src)
        #dec_states:[num_layers * num_directions, b_s, n_h]
        dec_states = self.decoder.begin_state(enc_states)
        for i, t in enumerate(tgt.permute(1, 0)):
            if i == max_len - 1: #teacher forcing 结束
                break
            if i == 0:
                dec_inputs = t #teacher forcing 第一时间步 t=<sos>
            else:
                #teacher forcing 按概率选择当前时间步输入
                is_teacher = random.random() < teacher_forcing_ratio
                top1 = outputs[i - 1].data.max(1)[1]
                dec_inputs = t if is_teacher else top1
            #dec_inputs:[batch_size, 1]
            #dec_states:[num_layers * num_directions, b_s, n_h]
            #enc_outputs:[s_l, b_s, num_hidden]
            #dec_output:[batch_size, 1, vocab_size]
            #dec_states: -
            dec_inputs = dec_inputs.unsqueeze(1)
            #print("dec_inputs:{}".format(dec_inputs.size()))
            dec_output, dec_states = self.decoder(dec_inputs, dec_states,
                   enc_outputs)
            outputs[i] = dec_output
        #outputs:[batch_size, seq_len, vocab_size]
        outputs = outputs.permute(1, 0, 2)
        return outputs

    def decode(self, src, tgt, method='beam-search'):
        enc_outputs, enc_states = self.encoder(src)
        dec_states = self.decoder.begin_state(enc_states)
        #print("enc_outputs:{}".format(enc_outputs.size()))
        #print("enc_states:{}".format(enc_states.size()))
        if method == 'greedy-search':
             return self.greedy_decode(tgt, dec_states, enc_outputs)
        else:
             return self.beam_decode(tgt, dec_states, enc_outputs)

    def greedy_decode(self, tgt, dec_states, enc_outputs):
        batch_size =tgt.size(0)
        decoded_batch = torch.zeros(batch_size, cfg.max_seq_len)
        dec_input = tgt
        for t in range(cfg.max_seq_len):
            #print("#######")
            #print("dec_input:{}".format(dec_input.size()))
            #print("dec_states:{}".format(dec_states.size()))
            
            dec_output, dec_states = self.decoder(dec_input, dec_states,
                    enc_outputs)
            #print("dec_output:{}".format(dec_output.size()))
            #print("dec_states:{}".format(dec_states.size()))
            topv, topi = dec_output.data.topk(1)
            #print("topv:{}".format(topv.size()))
            #print("topi:{}".format(topi.size()))
            
            topi = topi.view(-1)
            #print("topi:{}".format(topi.size()))
            decoded_batch[:, t] = topi

            dec_input = topi.detach().view(-1)
            dec_input = dec_input.unsqueeze(1)
        #print("decoded_batch:{}".format(decoded_batch))
        return decoded_batch

    def beam_decode(self, tgt, dec_states, enc_outputs):
        beam_width = 10
        topk = 3
        decoded_batch = []
        #print("tgt:{}".format(tgt.size()))
        #print("dec_states:{}".format(dec_states.size()))
        #print("enc_outputs:{}".format(enc_outputs.size()))

        #decode sentence by sentence
        for idx in range(tgt.size(0)): #batch_szie
            dec_state = dec_states[:, idx, :].unsqueeze(1)
            enc_output = enc_outputs[:, idx, :].unsqueeze(1)

            dec_input = tgt[idx].unsqueeze(1)
            #print("dec_input:{}".format(dec_input.size()))
            #print("dec_input:{}".format(dec_input))

            #number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            #starting node - hidden state, previous node, word id, logp, length
            node = BeamSearchNode(dec_state, None, dec_input, 0, 1)

            nodes = PriorityQueue()

            #star the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            #start beam search
            while True:
                if qsize > 200: break

                score, n = nodes.get()
                dec_input = n.wordid
                dec_state = n.h

                if n.wordid.item() == 2 and n.prevNode != None: #结束符 eos
                    endnodes.append((score, n))
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue
                #print("dec_input:{}".format(dec_input.size()))
                #print("dec_state:{}".format(dec_state.size()))
                #print("enc_output:{}".format(enc_output.size()))
                dec_output, dec_state = self.decoder(dec_input, dec_state,
                        enc_output)

                log_prob, indexes = torch.topk(dec_output, beam_width)
                nextnodes = []
                for new_k in range(beam_width):
                    decoded_t = indexes[0][0][new_k].view(-1).unsqueeze(1)
                    log_p = log_prob[0][0][new_k].item()
                    node = BeamSearchNode(dec_state, n, decoded_t, n.logp +
                            log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                qsize += len(nextnodes) - 1
            #end beam search

            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())
                utterance = utterance[::-1]
                utterances.append(torch.LongTensor(utterance))
            decoded_batch.append(utterances)
        return decoded_batch

class BeamSearchNode(object):
    def __init__(self, hidden_state, previousNode, wordId, logProb, length):
        self.h = hidden_state
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        #带有惩罚参数,参考吴恩达的beam-search
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.leng < other.leng

    def __gt__(self, other):
        return self.leng > self.leng
