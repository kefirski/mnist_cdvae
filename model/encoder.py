import torch as t
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=5, stride=2, padding=2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(50, 80, kernel_size=4, stride=2, padding=2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(80, 100, kernel_size=2, stride=1, padding=0),
            nn.ELU(),

            # nn.Linear(784, 400),
            # nn.ELU(inplace=True),
            #
            # nn.Linear(400, 300),
            # nn.ELU(inplace=True),
            #
            # nn.Linear(300, 100),
            # nn.ELU(inplace=True),
        )

        self.hidden_to_mu = nn.Linear(100, 80)
        self.hidden_to_logvar = nn.Linear(100, 80)

    def forward(self, input):
        # input = input.view(-1, 28 * 28)

        hidden = self.conv(input).view(-1, 100)

        return self.hidden_to_mu(hidden), self.hidden_to_logvar(hidden)
