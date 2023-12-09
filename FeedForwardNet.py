import torch.nn as nn
import torch.nn.functional as fun


class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = fun.relu(x)
        x = self.layer2(x)
        return x
