import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as fun
import math
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.W0 = nn.Linear(d_model, d_model)

    def calculate_attention(self, Q, K, V, mask=None):
        score = torch.matmul(Q, K.mT)
        if mask is not None:
            score += mask
            
        score_softmax = fun.softmax(score, dim=-1)
        score_scaled = score_softmax/math.sqrt(self.d_k)

        attention_score = torch.matmul(score_scaled, V)
        return attention_score

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, d_model = q.size()
        Q = self.WQ(q).view(batch_size, -1, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)
        K = self.WK(k).view(batch_size, -1, self.num_heads, self.d_k)
        K = K.transpose(1, 2)
        V = self.WV(v).view(batch_size, -1, self.num_heads, self.d_k)
        V = V.transpose(1, 2)

        attention_score = self.calculate_attention(Q, K, V, mask)

        attention_score = attention_score.transpose(1, 2)
        attention_output = attention_score.contiguous().view(batch_size, -1, self.d_model)

        return self.W0(attention_output)


def create_masks(src, target):
    single_mask = (src == 0).unsqueeze(1).unsqueeze(2)
    double_mask = (target == 0).unsqueeze(1).unsqueeze(3)
    length = target.size(1)

    look_ahead_mask = torch.triu(torch.ones(1, length, length), diagonal=1).bool()
    double_mask = double_mask & look_ahead_mask
    return single_mask, double_mask


def create_padding_mask(seq):
    return (seq == 0).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    return torch.triu(torch.ones((size, size)), diagonal=1).bool()

