import torch
import torch.nn as nn
import torch.nn.functional as fun
from EncoderBlock import EncoderBlock
from PositionalEncoding import PositionalEncoding


class SentimentModel(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len, num_classes, vocab_size):
        super(SentimentModel, self).__init__()
        self.encoder = EncoderBlock(d_model, d_ff, num_heads, max_seq_len)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.ff_net = nn.Linear(d_model, num_classes)

    def forward(self, x):
        encoder_input = self.positional_encoding(self.embedding(x))
        encoder_output = self.encoder(encoder_input)
        aggregated_output = torch.mean(encoder_output, dim=1)
        net_output = self.ff_net(aggregated_output)
        return net_output
