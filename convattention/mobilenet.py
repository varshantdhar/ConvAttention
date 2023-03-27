from site import USER_SITE
from urllib.parse import uses_netloc
import torch.nn as nn
import torch.nn.init

from common import conv1x1_block, conv3x3_dw_block, conv5x5_dw_block, SEUnit, Classifier



class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.conv_dw = conv3x3_dw_block(channels=in_channels, stride=stride)
        self.conv_pw = conv1x1_block(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv_dw(x)
        z = self.conv_pw(x)
        return z


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
                stride, activation="relu6", kernel_size=3, use_se=False) -> None:
        super().__init__()
        self.use_res_skip = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se

        self.conv1 = conv1x1_block(in_channels=in_channels, out_channels=mid_channels, 
                                    activation=activation)
        if kernel_size == 3:
            self.conv2 = conv3x3_dw_block(channels=mid_channels, stride=stride, activation=activation)
        elif kernel_size == 5:
            self.conv2 = conv5x5_dw_block(channels=mid_channels, stride=stride, activation=activation)
        else:
            raise ValueError
        if self.use_se:
            self.se_unit = SEUnit(channels=mid_channels, squeeze_factor=4, squeeze_activation="relu", excite_activation="hsigmoid")
        self.conv3 = conv1x1_block(in_channels=mid_channels, out_channels=out_channels, activation=None)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        if self.use_se:
            z = self.se_unit(z)
        z = self.conv3(z)
        if self.use_res_skip:
            z = z + x
        return x




