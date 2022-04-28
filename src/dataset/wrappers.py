"""
Wrappers for the different kinds of training settings we want to use
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


class Wrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    @property
    def n_factors(self):
        return self.base_dataset.n_factors

    @property
    def factors(self):
        return self.base_dataset.factors

    @property
    def images(self):
        return self.base_dataset.images

    @property
    def factor_values(self):
        return self.base_dataset.factor_values

    @property
    def factor_classes(self):
        return self.base_dataset.factor_classes

    @property
    def img_size(self):
        return self.base_dataset.img_size

    @property
    def factor_sizes(self):
        return self.base_dataset.factor_sizes

    @property
    def unique_values(self):
        return self.base_dataset.unique_values

    @property
    def transform(self):
        return self.base_dataset.transform

    @property
    def target_transform(self):
        return self.base_dataset.target_transform



def feature_set(dataset):
    n_values = [len(v) for v in dataset.unique_values.values()]

    sets =  []

    for j in range(dataset.n_factors):
        if dataset.categorical[j]:
            v = torch.from_numpy(dataset.factor_classes[..., j])
            one_hot = F.one_hot(v.to(dtype=torch.long), n_values[j]).numpy()
        else:
            one_hot = (dataset.factor_classes[..., j] / n_values[j])[..., None]

        sets.append(one_hot)

    return np.concatenate(sets, axis=-1)


class Supervised(Wrapper):
    def __init__(self, base_dataset, dim=None, pred_type='reg', norm_lats=True):
        super().__init__(base_dataset)
        self.pred_type = pred_type
        self.dim = dim
        self.norm_lats = norm_lats

        if norm_lats:
            mean_values, min_values, max_values = [], [], []
            for f in base_dataset.factors:
                mean_values.append(base_dataset.unique_values[f].mean())
                min_values.append(base_dataset.unique_values[f].min())
                max_values.append(base_dataset.unique_values[f].max())

            self._mean_values = np.array(mean_values)
            self._min_values = np.array(min_values)
            self._max_values = np.array(max_values)

            def standarize(factor_values):
                return ((factor_values - self._mean_values) /
                         (self._max_values - self._min_values))

            self.standarize = standarize

        if pred_type == 'set':
            self.sets = feature_set(self.base_dataset)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])

        if self.pred_type == 'class':
            target = self.factor_classes[idx]
        elif self.pred_type == 'set':
            target = self.sets[idx]
        else:
            target = self.factor_values[idx]

        if self.target_transform:
            target = self.target_transform(target)

        if self.dim is not None:
            target = target[self.dim]

        return img, target

    def __str__(self):
        return 'Supervised{}'.format(str(self.base_dataset))

    @property
    def n_targets(self):
        if self.pred_type == 'class':
            if self.dim is not None:
                return self.factor_sizes[self.dim]
            else:
                return self.factor_sizes
        elif self.pred_type == 'set':
            return self.sets.shape[-1]
        if self.dim is not None:
            return 1
        return self.n_factors


class Unsupervised(Wrapper):
    def __getitem__(self, idx):
        img = self.images[idx]

        if self.target_transform:
            img = self.transform(img)
            target = self.target_transform(img)
            return img, target

        img = self.transform(img)

        return img, img

    def __str__(self):
        return 'Unsupervised{}'.format(str(self.base_dataset))
