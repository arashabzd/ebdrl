import torch
import torch.nn as nn

from .utils import *


class ConvNet(nn.Module):
    def __init__(self, in_channels, width=1, init_mode=None):
        super().__init__()
        w1 = 2**(width+4)
        w2 = 2**(width+5)
        self.encoder = nn.Sequential(
            conv4x4(in_channels, w1, stride=2, padding=1),
            Swish(),
            conv4x4(w1, w1, stride=2, padding=1),
            Swish(),
            conv4x4(w1, w2, stride=2, padding=1),
            Swish(),
            conv4x4(w2, w2, stride=2, padding=1),
            Swish(),
            nn.Flatten(),
            linear(w2*4*4, 256)
        )

        if init_mode is not None:
            self.weight_init(init_mode)
    
    def weight_init(self, init_mode):
        if init_mode == 'kaiming':
            initializer = kaiming_init
        elif init_mode == 'normal':
            initializer = normal_init
        self.apply(initializer)

    def forward(self, x):
        return self.encoder(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=2 if downsample else 1, padding=1),
            Swish(),
            conv3x3(out_channels, out_channels, stride=1, padding=1)
        )
        self.res = nn.Sequential(
            nn.AvgPool2d(2) if downsample else nn.Identity(),
            conv1x1(in_channels, out_channels, stride=1, padding=0) if in_channels!=out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.conv(x)
        res = self.res(x)
        return swish(out+res)

class ResNet(nn.Module):
    def __init__(self, in_channels, width=1,  init_mode=None):
        super().__init__()
        w1 = 2**(width+4)
        w2 = 2**(width+5)
        self.encoder = nn.Sequential(
            conv3x3(in_channels, w1, stride=2, padding=1),
            ResBlock(w1, w1, True),
            ResBlock(w1, w2, True),
            ResBlock(w2, w2, True),
            nn.Flatten(),
            linear(w2*4*4, 256)
        )
        if init_mode is not None:
            self.weight_init(init_mode)
    
    def weight_init(self, init_mode):
        if init_mode == 'kaiming':
            initializer = kaiming_init
        elif init_mode == 'normal':
            initializer = normal_init
        self.apply(initializer)

    def forward(self, x):
        return self.encoder(x)
