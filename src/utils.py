import os
from collections import deque

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data


def export_model(model, path, input_shape):
    model = model.cpu().eval()
    traced_model = torch.jit.trace(model, torch.zeros(*input_shape), check_trace=False)
    torch.jit.save(traced_model, path)
    torch.save(model.state_dict(), path+'.statedict')
    return path

def import_model(path):
    model = torch.jit.load(path)
    model = model.cpu().eval()
    return model

def make_representor(model, device):
    model = model.to(device).eval()
    def represent(x):
        x = np.moveaxis(x, 3, 1)
        x = torch.from_numpy(x).to(device)
        x = x.to(torch.float32)
        with torch.no_grad():
            z = model(x)
        return z.cpu().numpy()
    return represent


class DlibDataset(Dataset):
    def __init__(self, name, seed=0):
        self.dataset = get_named_ground_truth_data(name)
        self.random_state = np.random.RandomState(seed)
        self.len = 737280

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        x, y = self.sample(1)
        return x[0], y[0]
    
    def sample_factors(self, n):
        y = self.dataset.sample_factors(n, self.random_state)
        return torch.from_numpy(y)
    
    def sample_observations_from_factors(self, factors):
        y = factors.numpy()
        x = self.dataset.sample_observations_from_factors(y, self.random_state)
        x = torch.from_numpy(np.moveaxis(x, 3, 1))
        return x.to(torch.float32)
    
    def sample(self, n):
        y = self.sample_factors(n)
        x = self.sample_observations_from_factors(y)
        return x, y

class Buffer:
    def __init__(self, p=.95, max_len=10000):
        self.p = p
        self.buffer = deque(maxlen=max_len)

    def update(self, x):
        for xi in x:
            self.buffer.append(xi)

    def sample(self, n):
        shape = [n] + list(self.buffer[0].shape)
        x = torch.rand(*shape)
        
        k = np.random.binomial(n, self.p)
        idx = np.random.choice(
            range(len(self.buffer)),
            size=k, 
            replace=False
        )
        
        for i, j in enumerate(idx):
            x[i] = self.buffer[j]
        return x
