import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.to_hidden = nn.Sequential(
            nn.Linear(784, 600),
            nn.ReLU(),

            nn.Linear(600, 400),
            nn.ReLU(),

            nn.Linear(400, 100),
            nn.ReLU()
        )

        self.hidden_to_mu = nn.Linear(100, 20)
        self.hidden_to_logvar = nn.Linear(100, 20)

    def forward(self, input):
        hidden = self.to_hidden(input.view(-1, 28 * 28))
        return self.hidden_to_mu(hidden), self.hidden_to_logvar(hidden)
