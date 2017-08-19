import torch as t
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()

        )

        self.hidden_to_mu = nn.Linear(400, 20)
        self.hidden_to_logvar = nn.Linear(400, 20)

    def forward(self, input):
        hidden = self.conv(input.view(-1, 28 * 28))
        return self.hidden_to_mu(hidden), self.hidden_to_logvar(hidden)
