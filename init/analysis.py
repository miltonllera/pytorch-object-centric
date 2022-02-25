import sys
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sacred import Ingredient
from skimage import color

from torch.utils.data.dataloader import DataLoader
from ignite.engine import create_supervised_evaluator


torch.set_grad_enabled(False)
sns.set(color_codes=True)
sns.set_style("white", {'axes.grid': False})
plt.rcParams.update({'font.size': 11})

analysis = Ingredient('analysis')


@analysis.capture
def infer(model, data, **kwargs):
    dataloader_args = {'batch_size': 128, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in loader:
            x = x.to(device=device)
            z = model(x)

            if isinstance(z, tuple):
                z = z[1]

            latents.append(z.cpu())
            targets.append(t)

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets


@analysis.capture
def model_score(model, data, metrics, model_name=None, device=None, *kwargs):
    dataloader_args = {'batch_size': 120, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    if model_name is None:
        model_name = 'model'
    if device is None:
        device = next(model.parameters()).device

    engine = create_supervised_evaluator(model, metrics, device)
    metrics = engine.run(loader).metrics

    index = pd.Index(metrics.keys(), name='Metric')
    scores = pd.Series(metrics.values(), index=index, name=model_name)

    return scores


@analysis.capture
def learning_curve_plot(model_folder, match='v2t', log_scale=True):
    path = os.path.join(model_folder, 'metrics.json')
    metrics = pd.read_json(path)

    data = []
    for _, m in metrics.items():
        steps, values = m['steps'], m['values']
        data.append(pd.DataFrame(zip(steps, values), columns=['iter', 'value']))

    if len(data) > 1:
        train_data = data[0]
        total_iters = len(train_data)
        n_epochs = len(data[1])
        iters_per_epoch = total_iters // n_epochs

        if match == 't2v':
            epoch = train_data['iter'] // iters_per_epoch
            train_data = train_data.groupby(epoch).mean()
            train_data['iter'] = np.arange(len(train_data))
            data[0] = train_data
        elif  match == 'v2t':
            for i in range(1, len(data)):
                val_data = data[i]
                val_data['iter'] = (val_data['iter'] + 1) * iters_per_epoch

    metrics = pd.concat(data, names=['metric'], keys=metrics.keys())
    metrics = metrics.reset_index()

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=metrics, x='iter', y='value', hue='metric', ax=ax)

    if log_scale:
        ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


@analysis.capture
def generate_recons(model, data, n_recons=10, loss='bce', **kwargs):
    dataloader_args = {'batch_size': n_recons, 'shuffle': True,
                       'pin_memory': True}
    dataloader_args.update(kwargs)

    inputs, targets = next(iter(DataLoader(data, **dataloader_args)))

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        recons = model(inputs.to(device=device))
        if isinstance(recons, tuple):
            recons = recons[0]

        if loss == 'bce':
            recons = recons.sigmoid()
        else:
            if recons.min() < 0.5:
                recons = recons / 2 + 0.5
            recons = recons.clamp(0, 1)

        recons = recons.cpu()

    return inputs, recons, targets


@analysis.capture
def plot_recons(data, no_recon_labels=False, axes=None):
    inputs, recons = data

    input_shape = inputs.shape[1:]
    batch_size = len(inputs)

    if axes is None:
        fig, axes = plt.subplots(2, batch_size, figsize=(2 * batch_size, 4))
    else:
        fig = None

    if inputs.min() < 0.0:
        inputs = inputs / 2 + 0.5

    images = np.stack([inputs.numpy(), recons.numpy()])

    for j, (examp_imgs, ylab) in enumerate(zip(images, ['original', 'recons'])):
        for i, img in enumerate(examp_imgs):
            if input_shape[0] == 3:
                axes[j, i].imshow(img.reshape(*input_shape).transpose(1, 2, 0))
            else:
                axes[j, i].imshow(img.reshape(*input_shape).transpose(1, 2, 0),
                                  cmap='Greys_r')

    for ax in axes.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if not no_recon_labels:
        axes[0, 0].set_ylabel('input', fontsize=20)
        axes[1, 0].set_ylabel('recons', fontsize=20)

    return fig


@analysis.capture
def compute_slot_masks(model, data, n_recons=10, **kwargs):
    dataloader_args = {'batch_size': n_recons, 'shuffle': True,
                       'pin_memory': True}
    dataloader_args.update(kwargs)

    inputs, targets = next(iter(DataLoader(data, **dataloader_args)))

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        z = model.embed(inputs.to(device=device))
        masks = model.decoder.masks(z).cpu()

    return inputs, masks, targets


def colorize(masks):
    n_slots, batch_size = masks.shape[:2]
    hue_rotations = np.linspace(0, 1, n_slots + 1)

    def tint(image, hue, saturation=1):
        hsv = color.rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return color.hsv2rgb(hsv)

    for i, h in zip(range(n_slots), hue_rotations):
        for j in range(batch_size):
            m = masks[i, j] > 1 / n_slots
            masks[i, j] = m * tint(masks[i, j], h)

    return masks


@analysis.capture
def plot_masks(data, no_recon_labels=False, axes=None):
    inputs, masks = data

    batch_size, n_slots = masks.shape[:2]
    img_shape = masks.shape[3:]

    inputs = inputs.unsqueeze_(0).permute(0, 1, 3, 4, 2).cpu().numpy()
    masks = masks.permute(1, 0, 3, 4, 2).cpu()
    masks = masks.repeat(1, 1, 1, 1, 3).numpy()

    masks = colorize(masks)
    if inputs.min() < 0.0:
        inputs = inputs / 2 + 0.5

    images = np.concatenate([inputs, masks])
    labels = ['original'] + ['slot {}'.format(str(i+1)) for i in range(n_slots)]

    if axes is None:
        fig, axes = plt.subplots(n_slots + 1, batch_size,
                                 figsize=(2 * batch_size, 4 * (n_slots + 1)))
    else:
        fig = None

    for j, examp_imgs in enumerate(images):
        if not no_recon_labels:
            axes[j, 0].set_ylabel(labels[j], fontsize=20)

        for i, img in enumerate(examp_imgs):
            axes[j, i].imshow(img.reshape(*img_shape, 3))

    for ax in axes.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return fig
