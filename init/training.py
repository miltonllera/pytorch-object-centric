"""
Sacred Ingredient for training functions.

The objective functions are defined and added as configureations to the
ingredient for ease of use. This allows chaging the objective function
easily and only needing to specify different parameters.
"""


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

import ignite.metrics as M
from ignite.engine import Events
from ignite.handlers import LRScheduler
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

from sacred import Ingredient

import src.training.loss as L
from src.training.scheduler import SmoothStepLR, WarmupAndDecay
from src.training.handlers import ModelCheckpoint


def binary_output(output):
    y_pred, y = output
    y_pred = (y_pred.sigmoid() > 0.5).to(dtype=y.dtype)
    return y_pred, y


bern_recons = {'name': 'recons_nll', 'params': {'loss': 'bce'}}
mse_recons  = {'name': 'recons_nll', 'params': {'loss': 'mse'}}
bxent_loss  = {'name': 'bxent', 'params': {}}
xent_loss   = {'name': 'xent', 'params': {}}
mse_loss    = {'name': 'mse', 'params': {}}
accuracy    = {'name': 'acc', 'params': {'output_transform': binary_output}}


training = Ingredient('training')

############################## dataloader ##############################

@training.capture
def init_loader(dataset, batch_size, train_val_split=0.0, **loader_kwargs):
    kwargs = {'shuffle': True, 'pin_memory': True, 'prefetch_factor': 2,
              'num_workers': 4, 'persistent_workers': False}
    kwargs.update(**loader_kwargs)

    num_workers = kwargs['num_workers']

    def wif(pid):
        process_seed = torch.initial_seed()
        base_seed = process_seed - pid

        sequence_seeder = np.random.SeedSequence([pid, base_seed])
        np.random.seed(sequence_seeder.generate_state(4))

    kwargs['pin_memory'] = kwargs['pin_memory'] and torch.cuda.is_available()

    if train_val_split > 0.0:
        val_length = int(len(dataset) * train_val_split)
        lenghts = [len(dataset) - val_length, val_length]

        train_data, val_data = random_split(dataset, lenghts)

        train_loader = DataLoader(train_data, batch_size, **kwargs,
                                  worker_init_fn=(wif if num_workers > 1
                                                  else None))
        val_loader = DataLoader(val_data, batch_size, **kwargs,
                                worker_init_fn=(wif if num_workers > 1
                                                else None))

        return train_loader, val_loader

    loader = DataLoader(dataset, batch_size, **kwargs,
                        worker_init_fn=(wif if num_workers > 1 else None))

    return loader, loader


@training.capture
def get_n_epochs(dataset_loader, n_iters, batch_size):
    data = dataset_loader.dataset
    total_instances = len(data)
    n_iters_per_epoch = int(np.ceil(total_instances / batch_size))
    return int(np.ceil(n_iters / n_iters_per_epoch))


############################## loss and metrics ##############################

@training.capture
def init_loss(loss):
    loss_fn, params = loss['name'], loss['params']
    if loss_fn == 'recons_nll':
        return L.ReconstructionNLL(**params)
    elif loss_fn == 'bxent':
        return nn.BCEWithLogitsLoss(**params)
    elif loss_fn == 'xent':
        return nn.CrossEntropyLoss(**params)
    elif loss_fn == 'mse':
        return nn.MSELoss(**params)
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn))


@training.capture
def init_metrics(metrics):
    metrics = list(map(dict.copy, metrics))

    labels = [m.pop('label', m['name']) for m in metrics]
    metrics = {l: get_metric(m) for (l, m) in zip(labels, metrics)}
    return metrics


@training.capture
def get_metric(metric):
    name = metric['name']
    params = metric['params']
    if name == 'mse':
        return M.MeanSquaredError(**params)
    elif name == 'recons_nll':
        return M.Loss(L.ReconstructionNLL(**params))
    elif name == 'bxent':
        return M.Loss(nn.BCEWithLogitsLoss(**params))
    elif name == 'xent':
        return M.Loss(nn.CrossEntropyLoss(**params))
    elif name == 'acc':
        return M.Accuracy(**params)
    raise ValueError('Unrecognized metric {}.'.format(metric))


################################ optimizers ###################################

@training.capture
def init_optimizer(optimizer, params, lr=0.01, l2_norm=0.0, **kwargs):

    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=l2_norm, **kwargs)
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(params, lr=lr, **kwargs)
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(params, lr=lr, weight_decay=l2_norm, **kwargs)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr,
                                  weight_decay=l2_norm, **kwargs)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr,
                              weight_decay=l2_norm, **kwargs)  # 0.01
    elif optimizer == 'nesterov':
        optimizer = optim.SGD(params, lr=lr, weight_decay=l2_norm,
                              nesterov=True, **kwargs)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, lr=lr,
                                  weight_decay=l2_norm, **kwargs)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(params, lr=lr,
                                   weight_decay=l2_norm, **kwargs)
    else:
        raise ValueError(r'Optimizer {0} not recognized'.format(optimizer))

    return optimizer


################################ schedulers ###################################

schedulers = {'plateau'     : training.capture(lr_scheduler.ReduceLROnPlateau),
              'exponential' : training.capture(lr_scheduler.ExponentialLR),
              'step'        : training.capture(lr_scheduler.StepLR),
              'smooth_step' : training.capture(SmoothStepLR),
              'warmup_decay': training.capture(WarmupAndDecay)}


@training.capture
def init_lr_scheduler(optimizer, scheduler, warmup=False, start_value=0.0,
                      warmup_steps=10000, end_value=None):
    if scheduler is None:
        return None

    scheduler = LRScheduler(schedulers[scheduler](optimizer))

    if warmup:  # for schedulers with no inbuilt warm-up phase
        scheduler = create_lr_scheduler_with_warmup(scheduler, start_value,
                                                    warmup_steps, end_value)
    return scheduler
