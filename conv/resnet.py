import enum
from numpy import block
from torch import norm
import torch.nn as nn
import torch.nn.init as nn_init

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        drop = self.dropout(x)
        out = drop + z
        out = self.relu(out)
        return out

class FullActivation(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()

        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        z = self.bn(x)
        z = self.relu(z)
        return z

class InitUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2) -> None:
        super().__init__()
        self.conv = conv7x7_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
    
    def forward(self, x):
        z = self.conv(x)
        z = self.pool(z)
        return z


class ResNet(nn.Module):
    def __init__(self, 
                channels, 
                num_classes, 
                in_size,
                init_unit_channels,
                block_type="orig",
                preact=False,
                in_channels=3) -> None:
        super().__init__()
        self.in_size = in_size
        self.preact = preact

        self.model = nn.Sequential()
        
        # normalize
        if preact:
            # BN + ReLU
            self.model.add_module("pre_activation", FullActivation(num_features=in_channels))
        else:
            # BN
            self.model.add_module("data_norm", nn.BatchNorm2d(num_features=in_channels))
        
        # init unit
        self.model.add_module("init_unit", InitUnit(in_channels=in_channels, out_channels=init_unit_channels))

        # stages
        in_channels = init_unit_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = 2 if (unit_id == 0) and (stage_id != 0) else 1
                if block_type == "orig":
                    stage.add_module(f"unit{unit_id + 1}", Original(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                elif block_type == "const_scaling":
                    stage.add_module(f"unit{unit_id + 1}", ConstScaling(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                elif block_type == "exclusive_gating":
                    stage.add_module(f"unit{unit_id + 1}", ExclusiveGating(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                elif block_type == "shortcut_gating":
                    stage.add_module(f"unit{unit_id + 1}", ShortcutGating(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                elif block_type == "conv_shortcut":
                    stage.add_module(f"unit{unit_id + 1}", ConvShortcut(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                elif block_type == "dropout_shortcut":
                    stage.add_module(f"unit{unit_id + 1}", DropoutShortcut(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                else:
                    raise ValueError('Invalid ResNet block module')
                in_channels = unit_channels
            self.model.add_module(f"stage{stage_id + 1}", stage)
        if preact:
            self.model.add_module("post_activation", FullActivation(num_features=in_channels))
        self.model.add_module("avg_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                nn_init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn_init.constant_(module.bias, 0)
        
        self.classifier.init_params()

    def forward(self, x):
        z = self.model(x)
        z = self.classifier(z)
        return z

        
        






