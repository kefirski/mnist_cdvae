import torch as t
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(16, 10, 2, 1, 0, bias=False),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(10, 5, 6, 2, 1, bias=False),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(5, 3, 6, 2, 1, bias=False),
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(3, 1, 6, 2, 2, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        input = input.unsqueeze(2).unsqueeze(2)
        return self.conv(input)
