import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),

            nn.Linear(400, 600),
            nn.ReLU(),

            nn.Linear(600, 784),
        )

    def forward(self, input):
        return self.fc(input).view(-1, 1, 28, 28)