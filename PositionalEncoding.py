import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div)
        pos_encoding[:, 1::2] = torch.cos(position * div)

        self.register_buffer('pe', pos_encoding.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)

        pos_encoding = self.pe[:, :seq_len]

        pos_encoding = pos_encoding.expand(x.size(0), -1, -1)
        return x + pos_encoding





