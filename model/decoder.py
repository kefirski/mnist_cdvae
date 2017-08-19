import torch as t
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(

            nn.Linear(20, 400),
            nn.ReLU(inplace=True),

            nn.Linear(400, 784),
            nn.Sigmoid()

        )

    def forward(self, input):
        result = self.conv(input)
        return result.view(-1, 1, 28, 28)
