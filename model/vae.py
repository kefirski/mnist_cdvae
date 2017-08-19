import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.encoder import Encoder
from model.decoder import Decoder


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, z=None):
        """
        :param input: an Float tensor with shape of [batch_size, 1, 28, 28]
        :param z: an Float tensor with shape of [batch_size, latent_size] if sampling is performed
        :return: an Float tensor with shape of [batch_size, 1, 28, 28], [batch_size, 16], [batch_size, 16]
        """

        mu = logvar = None
        if z is None:
            mu, logvar = self.decoder(input)

            [batch_size, latent_size] = mu.size()

            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, latent_size]))
            if input.is_cuda:
                z = z.cuda()

            z = z * std + mu

        return self.decoder(z), mu, logvar

    @staticmethod
    def divirgence_with_prior(mu, logvar):
        return (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()

    @staticmethod
    def divirgence_with_posterior(p_first, p_second):
        """
        :params p_first, p_second: tuple with parameters of distribution over latent variables
        :return: divirgence estimation
        """

        return 0.5 * t.sum(p_second[1] - p_first[1] + t.exp(p_first[1]) / (t.exp(p_second[1]) + 1e-8) +
                           t.pow(p_second[0] - p_second[0], 2) / (t.exp(p_second[1]) + 1e-8) - 1).mean()