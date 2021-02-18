import random
import pathlib
import argparse
import logging

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, distances

from .model import EBM
from .augmentation import Augmentation, dsprites_transforms
from .evaluation import compute_metrics
from . import utils


def train(args):
    if args.dataset == 'dsprites_full':
        transforms = dsprites_transforms
        in_channels = 1
    elif args.dataset == 'color_dsprites':
        transforms = dsprites_transforms
        in_channels = 3
    elif args.dataset == 'scream_dsprites':
        transforms = dsprites_transforms
        in_channels = 3
        
    logger.debug(f'loading dataset: {args.dataset}')
    dataset = utils.DlibDataset(
        args.dataset, 
        seed=args.seed
    )
    
    logger.debug(f'initializing dataloader')
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True if args.cuda else False,
        drop_last=True
    )
    
    augment = Augmentation(transforms)
    
    logger.debug(f'initializing model: {args.model}')
    model = EBM(
        encoder=args.model,
        in_channels=in_channels,
        n_factors=augment.n_factors,
        d_per_factor=args.dpf,
        free_dim=args.free_dim,
        encoder_head=args.encoder_head,
        energy_head=args.energy_head,
        init_mode=args.init_mode
    )
    logger.debug(f'copying model to: {device}')
    model = model.to(device)
    
    distance = distances.LpDistance(
        p=2, 
        power=2, 
        normalize_embeddings=False
    )
    contrastive_loss = losses.ContrastiveLoss(
        distance=distance
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=args.betas
    )
    
    logger.debug(f'initializing buffer: {args.buffer_size}')
    buffer = utils.Buffer(
        p=.95, 
        max_len=args.buffer_size
    )
    buffer.update(torch.rand(args.buffer_size, in_channels, 64, 64))
    
    logger.debug(f'training:')
    step = 0
    model.train()
    while step < args.steps:
        for x, _ in dataloader:
            x1, x2, f = augment(x)
            x = torch.cat([x1, x2]).to(device)
            z_pos, e_pos = model.get_energy(x)
            
            if args.divergence_weight > 0:
                u = buffer.sample(x.shape[0]).to(device)
                u = model.sample(
                    u, 
                    k=args.langevin_steps, 
                    noise=args.langevin_noise,
                    lr=args.langevin_lr
                )
                buffer.update(u.cpu())
                z_neg, e_neg = model.get_energy(u)
                loss_divergence = torch.mean(e_pos - e_neg)
                
                decay_pos = torch.clamp(e_pos.pow(2) - 1, min=0)
                decay_neg = torch.clamp(e_neg.pow(2) - 1, min=0)
                loss_energy_decay = torch.mean(decay_pos + decay_neg)
            else:
                loss_divergence = torch.zeros(1).to(device)
                loss_energy_decay = torch.zeros(1).to(device)
            
            
            y = torch.arange(f.shape[0])
            f = torch.cat([f, f])
            y = torch.cat([y, y])
            loss_contrastive = torch.zeros_like(loss_divergence)
            for j in range(augment.n_factors):
                z_j = z_pos[f==j, args.dpf*j:args.dpf*j+args.dpf]
                y_j = y[f==j]
                loss_contrastive += contrastive_loss(z_j, y_j)
            
            loss = args.divergence_weight*loss_divergence + args.contrastive_weight*loss_contrastive + args.energy_decay*loss_energy_decay
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), .01)
            optimizer.step()
            step += 1
            
            if step % args.log_interval == 0:
                log = (
                    f'Iteration {step}: loss={loss.item():.5f}, '
                    f'loss_contrastive={loss_contrastive.item():.5f}, '
                    f'loss_divergence={loss_divergence.item():.5f}, '
                    f'loss_energy_decay={loss_energy_decay.item():.5f}'
                )
                logger.info(log)
                
            if step >= args.steps:
                break
    logger.debug(f'exporting model: {model_path}')
    utils.export_model(model, model_path, input_shape=(1, in_channels, 64, 64))


def evaluate(args):
    args.metrics = None if args.metrics == '' else args.metrics.split()
    compute_metrics(model_path=model_path, 
                    output_dir=evaluation_dir, 
                    dataset_name=args.dataset, 
                    device=device, 
                    seed=args.seed,
                    metrics=args.metrics)
    
parser = argparse.ArgumentParser()

# general parameters
parser.add_argument('--config', 
                    type=str, default=None, 
                    help='Config file containing lines of commannd-line arguments (default: None).')
parser.add_argument('--experiment', 
                    type=str, default='debug', 
                    help='Experiment name (default: "debug").')
parser.add_argument('--model', 
                    type=str, default='convnet', 
                    help='Model name (default: "convnet").')
parser.add_argument('--dataset', 
                    type=str, default='dsprites_full', 
                    help='Dataset name (default: "dsprites_full").')
parser.add_argument('--seed', 
                    type=int, default=0,
                    help='Random seed (default: 0).')
parser.add_argument('--cuda',
                    action='store_true', default=False,
                    help='Enable CUDA.')

subparsers = parser.add_subparsers()
train_parser = subparsers.add_parser(
    'train',
    help='Train a model.'
)
# model parameters
train_parser.add_argument('--dpf',  
                          type=int, default=2, 
                          help='Dimension per factor (default: 2).')
train_parser.add_argument('--free-dim',  
                          type=int, default=0, 
                          help='Number of free dimensions of representation (default: 0).')
train_parser.add_argument('--encoder-head',  default=[],
                          type=int, nargs='+', 
                          help='Encoder head hidden layers as a list of ints (default: []).')
train_parser.add_argument('--energy-head',  default=[1000, 1000, 1000],
                          type=int, nargs='+', 
                          help='Energy head hidden layers as a list of ints (default: []).')
train_parser.add_argument('--init-mode', 
                          type=str, default=None, 
                          help='Weight initialization (default: None).')
# training parameters
train_parser.add_argument('--batch-size',  
                          type=int, default=64, 
                          help='Batch size (default: 64).')
train_parser.add_argument('--steps',  
                          type=int, default=100000, 
                          help='Number of training steps (iterations) (default: 100000).')
train_parser.add_argument('--lr',
                          type=float, default=0.0001,
                          help='Learning rate (default: 0.0001).')
train_parser.add_argument('--betas',  default=[0.9, 0.999],
                          type=float, nargs='+', 
                          help='Beta parameters of Adam optimizer (default: [0.9, 0.999]).')

train_parser.add_argument('--buffer-size',  
                          type=int, default=10000, 
                          help='Buffer size (default: 10000).')
train_parser.add_argument('--langevin-steps',  
                          type=int, default=60, 
                          help='Number of lagevin steps (default: 60).')
train_parser.add_argument('--langevin-noise',
                          type=float, default=0.005,
                          help='Langevin noise (default: 0.005).')
train_parser.add_argument('--langevin-lr',
                          type=float, default=10.0,
                          help='Langevin step size (default: 10.0).')
train_parser.add_argument('--divergence-weight',
                          type=float, default=1.0,
                          help='Weight of the contrastive divergence loss (default: 1.0).')
train_parser.add_argument('--contrastive-weight',
                          type=float, default=0.1,
                          help='Weight of the contrastive loss (default: 0.1).')
train_parser.add_argument('--energy-decay',
                          type=float, default=0.1,
                          help='Energy decay (default: 0.1).')
# other
train_parser.add_argument('--log-interval', 
                          type=int, default=100,
                          help='Tensorboard log interval (default: 100).')
train_parser.set_defaults(func=train)

eval_parser = subparsers.add_parser(
    'evaluate',
    help='Evaluate a model.'
)
eval_parser.add_argument('--metrics', 
                         type=str, default='', 
                         help='List of metrics (default: All available metrics).')
eval_parser.set_defaults(func=evaluate)


def run(args):
    global evaluation_dir, samples_dir, model_path, log_path
    result_dir = pathlib.Path('./results/')/args.experiment/args.model/args.dataset/str(args.seed)
    model_dir = result_dir/'saved_model'
    evaluation_dir = result_dir/'evaluation'
    samples_dir = result_dir/'samples'
    log_dir = result_dir/'logs'

    model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
     
    evaluation_dir = str(evaluation_dir)
    samples_dir = str(samples_dir)
    model_path = str(model_dir/'model.pt')
    log_path = str(log_dir/(args.func.__name__+'.log')) 

    global logger
    logger = logging.getLogger(args.experiment)
    fhandler = logging.FileHandler(filename=log_path, mode='w')
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    global device
    device = torch.device("cuda" if args.cuda else "cpu")

    logger.debug(args)
    args.func(args)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        with open(args.config) as config_file:
            configs = [line.split() for line in config_file.read().splitlines() if line]
        for config in configs:
            args = parser.parse_args(config)
            run(args)
    else:
        run(args)
