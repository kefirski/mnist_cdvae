import argparse, os, imageio, numpy as np
import torch as t, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms, torchvision.utils as vutils
from os import listdir
from make_grid import make_grid
from torchvision import datasets
from torch.optim import Adam
from model.vae import VAE
from torch.autograd import Variable

if __name__ == "__main__":

    if not os.path.exists('prior_sampling'):
        os.mkdir('prior_sampling')

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-epochs', type=int, default=40, metavar='NI',
                        help='num epochs (default: 40)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='batch size (default: 100)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save', type=str, default='trained_prior_vae', metavar='TS',
                        help='path where save trained model to (default: "trained_prior_vae")')
    args = parser.parse_args()

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vae = VAE()
    if args.use_cuda:
        vae = vae.cuda()

    optimizer = Adam(vae.parameters(), args.learning_rate, eps=1e-6)

    likelihood_function = nn.BCEWithLogitsLoss(size_average=False)

    z = Variable(t.randn(256, 20))
    if args.use_cuda:
        z = z.cuda()

    for epoch in range(args.num_epochs):
        for iteration, (input, _) in enumerate(dataloader):
            input = Variable(input)

            if args.use_cuda:
                input = input.cuda()

            optimizer.zero_grad()

            out, mu, logvar = vae(input)

            likelihood = likelihood_function(out, input) / args.batch_size
            loss = likelihood + VAE.divirgence_with_prior(mu, logvar)

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print('epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))

                sampling, _, _ = vae(input=None, z=z)

                grid = make_grid(F.sigmoid(sampling).cpu().data, 16, 28)
                vutils.save_image(grid, 'prior_sampling/vae_{}.png'.format(epoch * len(dataloader) + iteration))

    samplings = [f for f in listdir('prior_sampling')]
    samplings = [imageio.imread('prior_sampling/' + path) for path in samplings for _ in range(5)]
    imageio.mimsave('prior_sampling/movie.gif', samplings)

    t.save(vae.cpu().state_dict(), args.save)
