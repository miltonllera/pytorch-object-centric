"""
Sacred Ingredient for datasets

This ingredient has the functions to load datsets and DataLoaders for training.
Selecting a dataset is a matter of passing the corresponding name. There is a
function to get the splits, and one to show them (assuming they are iamges).

Three datasets are currently supported, dSprites, 3DShapes and MPI3D. The
transformation dataset can also be loaded using this function.
"""


import sys
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset, Subset as BaseSubset
from sacred import Ingredient

import init.configs.datasplits as splits
from init.training import init_loader

from src.dataset.shapes3d import load as load_shapes3d
from src.dataset.tetrominoes  import load as load_tetrominoes
import src.dataset.wrappers as data_wrappers

dataset = Ingredient('dataset')

# Datasets
load_shapes3d = dataset.capture(load_shapes3d)
load_tetrominoes = dataset.capture(load_tetrominoes)

# Wrappers
supervised = dataset.capture(data_wrappers.Supervised)
unsupervised = dataset.capture(data_wrappers.Unsupervised)
reconstruction = dataset.capture(data_wrappers.Reconstruction)

# Dataset configs
dataset.add_named_config('shapes3d', dataset='shapes3d')
dataset.add_named_config('tetrominoes', dataset='tetrominoes')

# Training setting configs
dataset.add_named_config('unsupervised', setting='unsupervised')
dataset.add_named_config('supervised', setting='supervised')


loaders  = {'shapes3d'      : load_shapes3d,
            'tetrominoes'   : load_tetrominoes}

wrappers = {'supervised'    : supervised,
            'reconstruction': reconstruction,
            'unsupervised'  : unsupervised}

splits   = {'shapes3d'      : splits.Shapes3D,
            'tetrominoes'   : splits.DummySplits}


@dataset.capture
def get_dataset(dataset, setting, data_filters, train=True):
    assert dataset in loaders, "Unrecognized dataset"
    assert setting in wrappers, "Unrecognized training setting"

    dataset = loaders[dataset](data_filters=data_filters, train=train)

    if setting is not None:
        dataset = wrappers[setting](dataset)

    return dataset


@dataset.capture
def get_data_spliters(dataset, condition=None, variant=None, modifiers=None):
    assert dataset in splits
    return splits[dataset].get_splits(condition, variant, modifiers)


@dataset.command(unobserved=True)
def plot():
    dataset = get_dataset(setting=None)
    loader = init_loader(dataset, 1, pin_memory=False,
                         shuffle=False, n_workers=1)

    for instance in loader:
        img = instance[0].reshape(
                loader.dataset.img_size).squeeze().numpy()

        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
            cmap = None
        else:
            cmap = 'Greys_r'

        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.show(block=True)


@dataset.capture
def get_lazyloader(dataset, condition=None, variant=None, modifiers=None):
    assert dataset in loaders, "Unrecognized dataset"

    # Mask with the modifiers only so that we have the full dataset
    modifier_mask = splits[dataset].get_splits(None, None, modifiers=modifiers)

    # Get the train/test masks separately, so they can be loaded as needed
    gen_maks = splits[dataset].get_splits(condition, variant)

    dataset = loaders[dataset](data_filters=modifier_mask)

    return DatasetLazyLoader(dataset, gen_maks)



class Subset(BaseSubset):
    @property
    def factor_values(self):
        return self.dataset.factor_values[self.indices]

    @property
    def factor_classes(self):
        return self.dataset.factor_classes[self.indices]

    @property
    def images(self):
        return self.dataset.images[self.indices]

    @property
    def factors(self):
        return self.dataset.factors

    @property
    def unique_values(self):
        return self.dataset.unique_values


@dataclass
class DatasetLazyLoader:
    dataset: Dataset
    partition_masks: object

    def get_unsupervised(self, train=True):
        return self.get_subset(data_wrappers.Unsupervised(self.dataset), train)

    def get_supervised(self, train=True, **kwargs):
        return self.get_subset(data_wrappers.Supervised(
            self.dataset, **kwargs), train)

    def get_reconstruction(self, train=True):
        return self.get_subset(data_wrappers.Reconstruction(self.dataset), train)

    def get_subset(self, dataset, train=True):
        mask = self.partition_masks[~train]

        if mask is None and not train:
            return None
        elif mask is None and train:
            return dataset

        idx = mask(self.dataset.factor_values,
                   self.dataset.factor_classes)

        return Subset(dataset, idx.nonzero()[0])

    @property
    def factors(self):
        return self.dataset.factors

    @property
    def n_factors(self):
        return self.dataset.n_factors

    @property
    def img_size(self):
        return self.dataset.img_size

    @property
    def factor_classes(self):
        return self.dataset

    @property
    def size(self):
        return len(self.dataset)

    def __len__(self):
        return len(self.dataset)
