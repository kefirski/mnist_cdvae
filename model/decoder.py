import torch as t
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(80, 10, 2, 1, 0, bias=False),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(60, 5, 6, 2, 1, bias=False),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(30, 3, 6, 2, 1, bias=False),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(30, 1, 6, 2, 2, bias=False),
            nn.Sigmoid()

            # nn.Linear(80, 100),
            # nn.ELU(inplace=True),
            #
            # nn.Linear(100, 300),
            # nn.ELU(inplace=True),
            #
            # nn.Linear(300, 500),
            # nn.ELU(inplace=True),
            #
            # nn.Linear(500, 784),
            # nn.Sigmoid()

        )

    def forward(self, input):
        input = input.unsqueeze(2).unsqueeze(2)
        return self.conv(input).view(-1, 28, 28)
