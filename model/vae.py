import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.encoder import Encoder
from model.decoder import Decoder
import math


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

        mu, logvar = self.encoder(input)

        [batch_size, latent_size] = mu.size()

        std = t.exp(0.5 * logvar)

        z = Variable(t.randn([batch_size, 15, latent_size]))
        if input.is_cuda:
            z = z.cuda()

        mu_repeated = mu.unsqueeze(1).repeat(1, 15, 1)
        std_repeated = std.unsqueeze(1).repeat(1, 15, 1)
        z = z * std_repeated + mu_repeated

        z = z.view(batch_size * 15, -1)

        return self.decoder(z), mu, logvar, z

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    @staticmethod
    def monte_carlo_divergence(z, mu, std, n):
        [batch_size, latent_size] = mu.size()

        log_p_z_x = VAE.normal_prob(z, mu, std)
        log_p_z = VAE.normal_prob(z,
                                  Variable(t.zeros(batch_size, latent_size)),
                                  Variable(t.ones(batch_size, latent_size)))

        result = log_p_z_x - log_p_z
        return result.view(-1, n).sum(1) / n

    @staticmethod
    def normal_prob(z, mu, std):
        return t.exp(-0.5 * ((z - mu) * t.pow(std + 1e-8, -1) * (z - mu)).sum(1)) / \
               t.sqrt(t.abs(2 * math.pi * std.prod(1)))

    @staticmethod
    def divergence_with_prior(mu, logvar):
        return (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()

    @staticmethod
    def divergence_with_posterior(p_first, p_second):
        """
        :params p_first, p_second: tuples with parameters of distribution over latent variables
        :return: divirgence estimation
        """

        return 0.5 * t.sum(2 * p_second[1] - 2 * p_first[1] + t.exp(p_first[1]) / (t.exp(p_second[1]) + 1e-8) +
                           t.pow(p_second[0] - p_second[0], 2) / (t.exp(p_second[1]) + 1e-8) - 1).mean()
