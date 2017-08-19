import torch as t
from torchvision import datasets
import torchvision.transforms as transforms
from model.decoder import Decoder
from torch.autograd import Variable

if __name__ == '__main__':
    # dataset = datasets.MNIST(root='data/',
    #                          transform=transforms.Compose([
    #                              transforms.ToTensor()]),
    #                          download=True)
    # dataloader = t.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    decoder = Decoder()

    x = Variable(t.randn(1, 16, 1, 1))
    print(decoder(x))