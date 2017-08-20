import torch as t


def make_grid(tensor, number, size):
    tensor = t.transpose(tensor, 0, 1).contiguous().view(1, number, number * size, size)
    tensor = t.transpose(tensor, 1, 2).contiguous().view(1, number * size, number * size)

    return tensor
