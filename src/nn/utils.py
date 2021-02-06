import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm


def conv4x4(in_channels, out_channels, stride=1, padding=0):
    return spectral_norm(
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=4, 
            stride=stride,
            padding=padding
        )
    )

def conv3x3(in_channels, out_channels, stride=1, padding=0):
    return spectral_norm(
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride,
            padding=padding
        )
    )

def conv1x1(in_channels, out_channels, stride=1, padding=0):
    return spectral_norm(
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride,
            padding=padding
        )
    )

def linear(in_features, out_features, sn=True):
    m = nn.Linear(
        in_features, 
        out_features
    )
    if sn:
        m = spectral_norm(m)
        
    return m

def swish(x):
    return x * torch.sigmoid(x)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight, 0, 1e-10)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)