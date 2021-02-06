import torch
import torch.nn as nn

from .utils import *


class MLP(nn.Module):
    def __init__(self, in_features, 
                 hidden_features, 
                 out_features, 
                 init_mode=None):
        super().__init__()
        features = [in_features] + hidden_features
        layers = []
        for i in range(len(features) - 1):
            layers.append(linear(features[i], features[i+1]))
            layers.append(Swish())
        layers.append(linear(features[-1], out_features, sn=False))
        self.mlp = nn.Sequential(*layers)
        
        if init_mode is not None:
            self.weight_init(init_mode)
    
    def weight_init(self, init_mode):
        if init_mode == 'kaiming':
            initializer = kaiming_init
        elif init_mode == 'normal':
            initializer = normal_init

        self.apply(initializer)
        
    def forward(self, x):
        return self.mlp(x)