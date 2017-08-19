import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input = nn.Linear(784, 128)

        self.hidden_to_mu = nn.Linear(128, 100)
        self.hidden_to_logvar = nn.Linear(128, 100)

    def forward(self, input):
        hidden = F.relu(self.input(input.view(-1, 28 * 28)))
        return self.hidden_to_mu(hidden), self.hidden_to_logvar(hidden)
