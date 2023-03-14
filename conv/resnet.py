import torch.nn as nn
import torch.nn.init

from common import conv1x1_block, conv3x3_block, conv7x7_block, Classifier

class Original(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)

    def forward(self, x):
        if self.use_projection:
            x = self.projection(x)
        z = self.conv1(x)
        z = self.conv2(z)
        z = z + x
        z = self.relu(z)
        return z

class ConstScaling(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.const = 0.5
        self.relu = nn.ReLU(inplace=True)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)

    def forward(self, x):
        if self.use_projection:
            x = self.projection(x)
        z = self.conv1(x)
        z = self.conv2(z)
        z = z * self.const
        z = z + (x * self.const)
        z = self.relu(z)
        return z

class ExclusiveGating(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        s = self.projection(x)
        p1 = z * s
        p2 = (1 - s) * x
        out = p1 + p2
        out = self.relu(out)
        return out

class ShortcutGating(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        s = self.projection(x)
        p1 = (1 - s) * x
        out = p1 + z
        return out

class ConvShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        s = self.projection(x)
        out = z + s
        out = self.relu(out)
        return out

class DropoutShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate=0.5) -> None:
        super().__init__()

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        drop = self.dropout(x)
        out = drop + z
        out = self.relu(out)
        return out

class FullPreActivation(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()

        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        z = self.bn(x)
        z = self.relu(z)
        return z


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()





