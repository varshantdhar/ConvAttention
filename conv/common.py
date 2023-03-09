from turtle import forward
import math

import torch.nn as nn
import torch.nn.functional as F
import torch


## activation functions

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x, inplace=True)

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6

class GELU(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(1.702 * x, inplace=True)


def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "relu6":
        return torch.nn.ReLU6(inplace=True)
    elif activation == "swish":
        return Swish()
    elif activation == "hswish":
        return HSwish()
    elif activation == "gelu":
        return GELU()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid(inplace=True)
    elif activation == "hsigmoid":
        return HSigmoid()
    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1,
        stride = stride,
        padding = 0,
        bias = bias
    )

def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        bias = bias
    )

def conv3x3_dw(channels, stride=1):
    return torch.nn.Conv2d(
        in_channels = channels,
        out_channels = channels,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        groups = channels,
        bias = False
    )

def conv5x5_dw(channels, stride=1):
    return torch.nn.Conv2d(
        in_channels = channels,
        out_channels = channels,
        kernel_size = 5,
        stride = stride,
        padding = 2,
        groups = channels,
        bias = False
    )