import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.input = nn.Linear(100, 128)
        self.out = nn.Linear(128, 784)

    def forward(self, input):
        hidden = F.relu(self.input(input))
        return F.sigmoid(self.out(hidden)).view(-1, 1, 28, 28)