import torch as t
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(

            nn.Linear(80, 300),
            nn.ELU(inplace=True),

            nn.Linear(300, 500),
            nn.ELU(inplace=True),

            nn.Linear(500, 784),
            nn.Sigmoid()

        )

    def forward(self, input):
        result = self.conv(input)
        return result.view(-1, 28, 28)
