import argparse
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets
from torch.optim import Adam
from model.vae import VAE
from torch.autograd import Variable
from tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter('vae/')

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='NI',
                        help='num epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='batch size (default: 100)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save', type=str, default=None, metavar='TS',
                        help='path where save trained model to (default: None)')
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

    likelihood_function = nn.MSELoss()
    # likelihood_function.size_average = False

    for epoch in range(args.num_epochs):
        for iteration, (input, _) in enumerate(dataloader):
            input = Variable(input)

            if args.use_cuda:
                input = input.cuda()

            optimizer.zero_grad()

            out, mu, logvar = vae(input)

            likelihood = likelihood_function(out, input)
            loss = likelihood + VAE.divirgence_with_prior(mu, logvar)

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:

                print('epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))

                z = Variable(t.randn(64, 20))
                if args.use_cuda:
                    z = z.cuda()

                sampling, _, _ = vae(input=None, z=z)

                sampling = vutils.make_grid(sampling.data, scale_each=True)
                writer.add_image('sampling', sampling, epoch * len(dataloader) + iteration)

    writer.close()
    t.save(vae.state_dict(), 'trained_VAE')

