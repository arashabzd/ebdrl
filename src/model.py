import torch
import torch.nn as nn
import torch.optim as optim

from .nn import *


class EBM(nn.Module):
    def __init__(self, encoder, in_channels, 
                 n_factors, d_per_factor, free_dim,
                 encoder_head=[1000, 1000, 1000],
                 energy_head=[],
                 init_mode=None):
        super().__init__()
        self.in_channels = in_channels
        self.n_factors = n_factors
        self.d_per_factor = d_per_factor
        
        if encoder == 'convnet':
            self.encoder = ConvNet(in_channels, width=1, init_mode=init_mode)
        elif encoder == 'wide-convnet':
            self.encoder = ConvNet(in_channels, width=2, init_mode=init_mode)
        elif encoder == 'wider-convnet':
            self.encoder = ConvNet(in_channels, width=3, init_mode=init_mode)
        elif encoder == 'resnet':
            self.encoder = ResNet(in_channels, width=1, init_mode=init_mode)
        elif encoder == 'wide-resnet':
            self.encoder = ResNet(in_channels, width=2, init_mode=init_mode)
        elif encoder == 'wider-resnet':
            self.encoder = ResNet(in_channels, width=3, init_mode=init_mode)
        
        z_dim = n_factors*d_per_factor + free_dim
        self.encoder_head = MLP(256, encoder_head, z_dim, init_mode) 
        self.energy_head = MLP(z_dim, energy_head, 1, init_mode)
    
    def get_representation(self, x):
        c = self.encoder(x)
        z = self.encoder_head(c)
        return z
    
    def forward(self, x):
        return self.get_representation(x)
    
    def get_energy(self, x):
        z = self.get_representation(x)
        e = self.energy_head(z)
        return z, e

    def sample(self, x,
               k=20, noise=.005, 
               lr=10, momentum=0., nesterov=False, 
               grad_clip=True):
        x.requires_grad_(True)
        optimizer = optim.SGD([x], lr=lr, momentum=momentum, nesterov=nesterov)
        for i in range(k):
            optimizer.zero_grad()
            self.get_energy(x)[1].sum().backward()
            if grad_clip:
                torch.nn.utils.clip_grad_value_([x], .01)
            optimizer.step()
            with torch.no_grad():
                x += noise * torch.randn_like(x)
                x.clamp_(0, 1)
        x.requires_grad_(False)
        return x
        