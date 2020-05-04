# Copyright (c) Microsoft. All rights reserved.
import torch
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .dropout_wrapper import DropoutWrapper
from .similarity import FlatSimilarityWrapper, SelfAttnWrapper, DualAttentionWrapper, AttentionWrapper
from .my_optim import weight_norm as WN
from .common import activation, init_wrapper

SMALL_POS_NUM=1.0e-30

def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training: dropout_p = 0.0
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1)-1)
        new_data[i][one] = 1
    mask = 1.0/(1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask


def masked_select(tensor, mask):
    mask_len = mask.sum(dim=-1)
    max_seq_len = mask_len.max()
    new_tensor = torch.zeros(tensor.size(0), max_seq_len, tensor.size(-1)).cuda()
    new_mask = torch.ones(tensor.size(0), max_seq_len).cuda()
    for i in range(tensor.size(0)):
        new_tensor[i, :mask_len[i]] = torch.masked_select(tensor[i], mask[i].unsqueeze(1).expand_as(tensor[i])).view(-1, tensor.size(-1))
        new_mask[i, :mask_len[i]] = 0
    return new_tensor, new_mask.byte()


class Classifier(nn.Module):
    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)

        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)

        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None, activation=None):
        seq_len = None
        if len(x1.size()) == 3:
            bz, seq_len, hidden_size = x1.size()
            x1 = x1.contiguous().view(-1, hidden_size)
            x2 = x2.contiguous().view(-1, hidden_size)

        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        if activation:
            scores = activation(self.proj(x))
        else:
            scores = self.proj(x)

        if seq_len:
            return scores.view(bz, seq_len, -1)
        else:
            return scores


class SANClassifier(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """
    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(SANClassifier, self).__init__()
        self.prefix = prefix
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn = getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        # self.hyp_attn = None
        # if opt.get('hyp_attn_premise', 0):
        #     self.hyp_attn = AttentionWrapper(x_size, h_size, prefix=prefix, opt=opt, dropout=self.dropout)
        #     self.hyp_merge = Classifier(x_size, x_size, opt, prefix=prefix, dropout=self.dropout)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(x_size, 1, opt, prefix=prefix, dropout=self.dropout)

    def forward(self, x, h0, x_mask=None, h_mask=None, is_training=True):
        # if self.hyp_attn:
        #     h_attn = self.hyp_attn(h0, x, key_padding_mask=x_mask)
        #     h0 = self.hyp_merge(h0, h_attn, activation=self.f)

        h0 = self.query_wsum(h0, h_mask)
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            # next turn
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            batch_size = x.size(0) // self.label_size
            mask = generate_mask(self.alpha.data.new(batch_size, self.num_turn), self.mem_random_drop, is_training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [mask[idx].view(batch_size, 1).expand_as(inp.view(-1, self.label_size))
                               * F.softmax(inp.view(-1, self.label_size), 1)
                               for idx, inp in enumerate(scores_list)]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores


class SANClassifier2(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """
    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(SANClassifier2, self).__init__()
        self.prefix = prefix
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.dual_attn = DualAttentionWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn = getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        self.hyp_first = opt.get('{}_hyp_first'.format(prefix), 1)
        self.hyp_raw = opt.get('{}_hyp_raw'.format(prefix), 0)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(x_size, 1, opt, prefix=prefix, dropout=self.dropout)

        self.premise_merge = Classifier(x_size, x_size, opt, prefix=prefix, dropout=self.dropout)
        self.hyp_merge = Classifier(x_size, x_size, opt, prefix=prefix, dropout=self.dropout)


    def forward(self, x, h, x_mask=None, h_mask=None, is_training=True):
        if self.hyp_first and self.hyp_raw:
            pass
        elif self.hyp_first and not self.hyp_raw:
            _, h_attn = self.dual_attn(x, h, x_mask, h_mask)

        # x_prime = self.premise_merge(x, x_attn, activation=self.f)
            h = self.hyp_merge(h, h_attn, activation=self.f)
        else:
            raise NotImplementedError

        # if self.num_turn == 0:
        #     scores = self.classifier(x_prime.max(dim=1)[0], h_prime.max(dim=1)[0])
        #     return scores

        # if self.hyp_first and not self.hyp_raw:
        #     # x = x_prime
        #     h = h_prime
        # elif self.hyp_first and self.hyp_raw:
        #     # x = x_prime
        #     pass
        # elif not self.hyp_first and not self.hyp_raw:
        #     x = h_prime
        #     h = x_prime
        # else:
        #     h = x
        #     x = h_prime

        h0 = self.query_wsum(h, h_mask)
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            # next turn
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            batch_size = x.size(0) // self.label_size
            mask = generate_mask(self.alpha.data.new(batch_size, self.num_turn), self.mem_random_drop, is_training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [mask[idx].view(batch_size, 1).expand_as(inp.view(-1, self.label_size))
                               * F.softmax(inp.view(-1, self.label_size), 1)
                               for idx, inp in enumerate(scores_list)]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores