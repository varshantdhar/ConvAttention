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

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels = 1,
                                    out_channels = num_classes,
                                    kernel_size = 1,
                                    bias = True)
        
        self.flatten = Flatten()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten.forward(x)
        return x

    def init_params(self):
        nn.init.xavier_normal_(self.conv.weight, gain = 1.0)

class SEUnit(nn.Module):
    def __init__(self,
                 channels,
                 squeeze_factor = 16,
                 squeeze_activation = "relu",
                 excite_activation = "sigmoid") -> None:
        super().__init__()
        squeeze_channels = channels // squeeze_factor
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels, out_channels=squeeze_channels, bias=True)
        self.activation1 = get_activation(squeeze_activation)
        self.conv2 = conv1x1(in_channels=squeeze_channels, out_channels=channels, bias=True)
        self.activation2 = get_activation(excite_activation)

    def forward(self, x):
        z = self.pool(x)
        z = self.conv1(z)
        z = self.activation1(z)
        z = self.conv2(z)
        s = self.activation2(z)
        return x * s


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1,
        stride = stride,
        padding = 0,
        bias = bias
    )

def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        bias = bias
    )

def conv3x3_dw(channels, stride=1):
    return nn.Conv2d(
        in_channels = channels,
        out_channels = channels,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        groups = channels,
        bias = False
    )

def conv5x5_dw(channels, stride=1):
    return nn.Conv2d(
        in_channels = channels,
        out_channels = channels,
        kernel_size = 5,
        stride = stride,
        padding = 2,
        groups = channels,
        bias = False
    )

class ConvBlock(nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups = 1,
                bias = False,
                use_bn = True,
                activation="relu") -> None:
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = (activation is not None)

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups,
            bias = bias
        )

        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.use_bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x

def conv1x1_block(in_channels,
                 out_channels,
                 stride=1,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
    return ConvBlock(in_channels = in_channels,
                     out_channels = out_channels,
                     kernel_size = 1,
                     stride = stride,
                     bias = bias,
                     padding = 0,
                     use_bn = use_bn,
                     activation = activation)

def conv3x3_block(in_channels,
                 out_channels,
                 stride=1,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
    return ConvBlock(in_channels = in_channels,
                     out_channels = out_channels,
                     kernel_size = 3,
                     stride = stride,
                     padding = 1,
                     stride = stride,
                     bias = bias,
                     use_bn = use_bn,
                     activation = activation)

def conv7x7_block(in_channels,
                 out_channels,
                 stride=1,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
    return ConvBlock(in_channels = in_channels,
                     out_channels = out_channels,
                     kernel_size = 7,
                     stride = stride,
                     padding = 3,
                     stride = stride,
                     bias = bias,
                     use_bn = use_bn,
                     activation = activation)

def conv3x3_dw_block(channels,
                    stride = 1,
                    bias = False,
                    use_bn = True,
                    activation = "relu"):
    return ConvBlock (in_channels = channels,
                     out_channels = channels,
                     kernel_size = 3,
                     groups = channels,
                     stride = stride,
                     bias = bias,
                     padding = 1,
                     use_bn = use_bn,
                     activation = activation
                     )

def conv5x5_dw_block(channels,
                    stride = 1,
                    bias = False,
                    use_bn = True,
                    activation = "relu"):
    return ConvBlock (in_channels = channels,
                     out_channels = channels,
                     kernel_size = 5,
                     groups = channels,
                     stride = stride,
                     bias = bias,
                     padding = 2,
                     use_bn = use_bn,
                     activation = activation
                     )