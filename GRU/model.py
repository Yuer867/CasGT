import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import Constants


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, opt, dropout=0.1, tie_weights=False):
        super(RNNModel, self).__init__()
        ntoken = opt.user_size
        ninp = opt.d_word_vec
        nhid = opt.d_inner_hid

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_emb = opt.pos_emb
        if self.pos_emb:
            self.pos_dim = 8
            self.pos_embedding = nn.Embedding(1000, self.pos_dim)

        if self.pos_emb:
            self.rnn = getattr(nn, rnn_type)(ninp + self.pos_dim, nhid)
        else:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.user_size = ntoken

    def neighbor_sampling(self, nodes, num_neighbor):
        sample = np.zeros((nodes.shape[0], num_neighbor), dtype=int)
        for i in range(nodes.shape[0]):
            sample[i, 0] = nodes[i]
            sample[i, 1:] = np.random.choice(self.adj_list[nodes[i]], num_neighbor - 1, replace=True)
        return sample

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()
        masked_seq = previous_mask * seqs.data.float()

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))

        return masked_seq

    def forward(self, input, generate=False):
        if not generate:
            input = input[:, :-1]
        user_emb = self.encoder(input)
        emb = self.drop(user_emb)
        batch_size = input.size(0)
        max_len = input.size(1)
        outputs = Variable(torch.zeros(max_len, batch_size, self.nhid)).cuda()
        hidden = Variable(torch.zeros(batch_size, self.nhid)).cuda()
        for t in range(0, max_len):
            # GRU
            if self.pos_emb:
                hidden = self.rnn(
                    torch.cat([emb[:, t, :], self.drop(self.pos_embedding(torch.ones(batch_size).long().cuda() * t))],
                              dim=1), hidden)
            else:
                hidden = self.rnn(emb[:, t, :], hidden)
            outputs[t] = hidden

        outputs = outputs.transpose(0, 1).contiguous()  # b*l*v
        outputs = self.drop(outputs)
        decoded = self.decoder(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2)))
        result = decoded.view(outputs.size(0), outputs.size(1), decoded.size(1)) + torch.autograd.Variable(
            self.get_previous_user_mask(input, self.user_size), requires_grad=False)

        return result.view(-1, decoded.size(1)), hidden, self.encoder.weight