import torch as t
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=2, padding=2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.ELU(),
        )

        self.hidden_to_mu = nn.Linear(12, 15)
        self.hidden_to_logvar = nn.Linear(12, 15)

    def forward(self, input):

        hidden = self.conv(input).view(-1, 12)

        return self.hidden_to_mu(hidden), self.hidden_to_logvar(hidden)