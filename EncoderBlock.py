import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNet import FeedForwardNet


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len):
        super(EncoderBlock, self).__init__()
        self.embedding = nn.Embedding(d_model, max_seq_len)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.multi_head_att = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff_net = FeedForwardNet(d_model, d_ff)

    def forward(self, x, mask=None):
        attention_output = self.multi_head_att(x, x, x, mask)
        norm1 = self.layer_norm1(x + attention_output)
        ff_output = self.ff_net(norm1)
        norm2 = self.layer_norm2(ff_output + norm1)
        return norm2
