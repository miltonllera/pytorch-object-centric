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
import ignite.metrics as M
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from sacred import Ingredient
from ignite.engine import _prepare_batch, Engine

import src.training.loss as L
from src.training.scheduler import WarmupLR, ExponentialLR
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
slate_loss  = {'name': 'slate', 'params': {'loss': 'mse'}}
token_xent  = {'name': 'token_xent', 'params': {}}
accuracy    = {'name': 'acc', 'params': {'output_transform': binary_output}}
hungarian_loss = {'name': 'hunloss', 'params': {}}


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
def create_supervised_trainer(model, optimizer, loss_fn, device=None,
                              non_blocking=False, grad_norm=None,
                              norm_type='inf',
                              prepare_batch=_prepare_batch,
                              output_transform=None) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    if output_transform is None:
        output_transform = lambda x, y, y_pred, loss: loss.item()

    def _update(engine, batch):
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        if grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm, norm_type)

        optimizer.step()

        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


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
    elif loss_fn == 'token_xent':
        return L.ImageTokenLoss(**params)
    elif loss_fn == 'slate':
        return L.SLATELoss()
    elif loss_fn == 'hunloss':
        return L.HungarianLoss()
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
    elif name == 'token_xent':
        return M.Loss(L.ImageTokenLoss(**params),
                output_transform=lambda output: output[0][-1])
    elif name == 'acc':
        return M.Accuracy(**params)
    elif name == 'hunloss':
        return M.Loss(L.HungarianLoss())
    raise ValueError('Unrecognized metric {}.'.format(name))


################################ optimizers ###################################

def param_group_idx(group_prefixes, param_name):
    for i, prefix in enumerate(group_prefixes):
        if prefix in param_name:
            return i
    return -1


@training.capture
def init_optimizer(optimizer, model, lr=0.01, l2_norm=0.0,
                   param_groups=None, **kwargs):
    if param_groups is None:
        params = model.parameters()
    else:
        group_prefix = [pg[0] for pg in param_groups]
        params = [{'params': [], **pg[1]} for pg in param_groups]
        other_params = []

        for p_name, p in model.named_parameters():
            idx = param_group_idx(group_prefix, p_name)
            if idx == -1:
                other_params.append(p)
            else:
                params[idx]['params'].append(p)

        if len(other_params) > 0:
            params.append({'params': other_params})

        assert (len(list(model.parameters())) ==
                sum(len(p['params']) for p in params))

    # params = [{'params': model.patch_ae.parameters(), 'lr': 0.0003},
    #           {'params': model.slot_attn.parameters()},
    #           {'params': model.gpt_decoder.parameters()},
    #           {'params': model.token2emb.parameters()},
    #           {'params': model.slot_proj.parameters()},
    #           {'params': model.emb2token.parameters()}]

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

plateau_scheduler =  training.capture(lr_scheduler.ReduceLROnPlateau)
schedulers = {'exponential' : training.capture(ExponentialLR),
              'warmup'      : training.capture(WarmupLR)}


@training.capture
def init_lr_scheduler(optimizer, scheduler, warmup=False, start_value=0.0,
                      warmup_steps=10000, end_value=None):
    if scheduler is not None:
        scheduler = schedulers[scheduler](optimizer)

    if warmup:
        warmup = schedulers['warmup'](optimizer)
        if scheduler is not None:
            scheduler = lr_scheduler.ChainedScheduler([scheduler, warmup])
        else:
            scheduler = warmup

    return scheduler


@training.capture
def reduce_on_plateau(optimizer, reduce_on_plateau=False):
    if not reduce_on_plateau:
        return None
    return plateau_scheduler(optimizer)
