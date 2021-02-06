import random

import torch
from torchvision import transforms, datasets


class RandomRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

    
class GaussianNoise:
    def __init__(self, std=.01):
        self.std = std
        
    def __call__(self, x):
        out = x + torch.randn_like(x)*self.std
        return out.clamp_(0, 1)
    

class Augmentation:
    def __init__(self, transforms):
        self.transforms = transforms
        self.n_factors = len(transforms)

    def augment_batch(self, x, f):
        return torch.stack(
            [
                self.transforms[f[i]](x[i]) 
                for i in range(x.shape[0])
            ]
        )
    
    def __call__(self, x):
        f = torch.randint(self.n_factors, size=(x.shape[0],))
        x1 = self.augment_batch(x, f)
        x2 = self.augment_batch(x, f)
        return x1, x2, f


dsprites_transforms = [
    # shape preserving
    transforms.Compose(
        [
            transforms.ToPILImage(),
            RandomRotation([0, 90, 180, 270]),
            transforms.Pad(16),
            transforms.RandomResizedCrop(64, scale=(.5, 1), ratio=(1., 1.)),
            transforms.ToTensor(),
        ]
    ),
    # scale preserving
    transforms.Compose(
        [
            transforms.ToPILImage(),
            RandomRotation([0 ,90, 180, 270]),
            transforms.RandomCrop((64, 64), padding=(16, 16, 16, 16)),
            transforms.ToTensor(),
        ]
    ),
    # orientation preserving
    transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Pad(16),
            transforms.RandomResizedCrop(64, scale=(.5, 1), ratio=(1., 1.)),
            transforms.ToTensor(),
        ]
    ),
    # x preserving
    transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop((64, 64), padding=(0, 16, 0, 16)),
            transforms.ToTensor(),
        ]
    ),
    # y preserving
    transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop((64, 64), padding=(16, 0, 16, 0)),
            transforms.ToTensor(),
        ]
    )
]
