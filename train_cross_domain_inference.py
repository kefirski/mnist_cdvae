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
    parser.add_argument('--batch-size', type=int, default=20, metavar='BS',
                        help='batch size (default: 20)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--save', type=str, default=None, metavar='TS',
                        help='path where save trained model to (default: None)')
    args = parser.parse_args()

    train_dataset = datasets.MNIST(root='data/',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()]),
                                   download=True,
                                   train=True)
    train_dataloader = t.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = datasets.MNIST(root='data/',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()]),
                                   download=True,
                                   train=False)
    valid_dataloader = t.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
