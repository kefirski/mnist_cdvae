import torch as t
from torchvision import datasets
import torchvision.transforms as transforms
from model.encoder import Encoder
from torch.autograd import Variable

if __name__ == '__main__':
    # dataset = datasets.MNIST(root='data/',
    #                          transform=transforms.Compose([
    #                              transforms.ToTensor()]),
    #                          download=True)
    # dataloader = t.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    encoder = Encoder()

    x = Variable(t.randn(1, 1, 28, 28))
    print(encoder(x))