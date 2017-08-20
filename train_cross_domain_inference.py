import argparse
import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from os import listdir
from make_grid import make_grid
from torchvision import datasets
from torch.optim import Adam
from model.vae import VAE
from torch.autograd import Variable


def reflect_tensor(tensor, axis):
    tensor = np.flip(tensor.numpy(), axis)
    return t.from_numpy(tensor.copy())


if __name__ == "__main__":

    if not os.path.exists('cross_domain_sampling'):
        os.mkdir('cross_domain_sampling')

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='NI',
                        help='num epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='batch size (default: 10)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save', type=str, default=None, metavar='TS',
                        help='path where save trained model to (default: None)')
    parser.add_argument('--saved', type=str, default='trained_prior_vae', metavar='TS',
                        help='path to saved prior model (default: "trained_prior_vae")')
    args = parser.parse_args()

    train_dataset = datasets.MNIST(root='data/',
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()]),
                                   train=True)
    dataloader = t.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = datasets.MNIST(root='data/',
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()]),
                                   train=False)

    noisy_vae = VAE()
    prior_vae = VAE()
    prior_vae.load_state_dict(t.load(args.saved))
    if args.use_cuda:
        noisy_vae, prior_vae = noisy_vae.cuda(), prior_vae.cuda()

    optimizer = Adam(noisy_vae.parameters(), args.learning_rate, eps=1e-6)

    likelihood_function = nn.BCEWithLogitsLoss(size_average=False)

    prior_valid = t.stack([valid_dataset[i][0] for i in range(256)])
    noisy_valid = t.stack([reflect_tensor(var, 2) for var in prior_valid])

    noisy_grid = make_grid(noisy_valid, 16, 28)
    vutils.save_image(noisy_grid, 'cross_domain_sampling/noised_data.png')

    prior_valid, noisy_valid = Variable(prior_valid), Variable(noisy_valid)
    if args.use_cuda:
        prior_valid, noisy_valid = prior_valid.cuda(), noisy_valid.cuda()

    for epoch in range(args.num_epochs):
        for iteration, (prior_input, _) in enumerate(dataloader):
            noisy_input = reflect_tensor(prior_input, 3)

            prior_input, noisy_input = Variable(prior_input), Variable(noisy_input)

            if args.use_cuda:
                prior_input, noisy_input = prior_input.cuda(), noisy_input.cuda()

            optimizer.zero_grad()

            noisy_out, noisy_mu, noisy_logvar = noisy_vae(noisy_input)
            prior_mu, prior_logvar = prior_vae.encode(prior_input)

            likelihood = likelihood_function(noisy_out, noisy_input) / args.batch_size
            loss = likelihood + VAE.divirgence_with_posterior((noisy_mu, noisy_logvar), (prior_mu, prior_logvar)) + VAE.divirgence_with_prior(noisy_mu, noisy_logvar)
            # loss = likelihood + VAE.divirgence_with_posterior((noisy_mu, noisy_logvar), (prior_mu, prior_logvar))

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print('epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))

                mu, logvar = noisy_vae.encode(noisy_valid)

                [batch_size, latent_size] = mu.size()

                std = t.exp(0.5 * logvar)

                z = Variable(t.randn([batch_size, latent_size]))
                if noisy_valid.is_cuda:
                    z = z.cuda()

                z = z * std + mu

                sampling = prior_vae.decode(z)

                grid = make_grid(F.sigmoid(sampling).cpu().data, 16, 28)
                vutils.save_image(grid, 'cross_domain_sampling/vae_{}.png'.format(epoch * len(dataloader) + iteration))
