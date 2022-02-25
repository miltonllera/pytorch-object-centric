"""
Data-splitting functions for each dataset.

These are the functions that exlcude combiantions from the datasets
in order to test different generalisation setttings. The splits are
organized in classes so they create different namespaces.

The general mechanism works by passing a condition and variant
parameter to the appropriate class. The splits are returned as index
values. That way the images and targets (when predicting a factor)
can be split in one call.

Each dataset contains a description of the generative factor names
and their values for quick referencing when adding more splits.
"""


import numpy as np
from functools import partial


def compose(mask, mod):
    def composed(factor_values, factor_classes):
        return (mask(factor_values, factor_classes) &
                mod(factor_values, factor_classes))
    return composed


class DataSplit:
    interp         = {}

    recomb2element = {}

    recomb2range   = {}

    extrp          = {}

    modifiers      = {}

    @classmethod
    def get_splits(cls, condition, variant, modifiers=None):
        try:
            if condition is None:
                masks = None, None
            elif condition == 'interp':
                masks = cls.interp[variant]()
            elif condition == 'recomb2element':
                masks = cls.recomb2element[variant]()
            elif condition == 'recomb2range':
                masks = cls.recomb2range[variant]()
            elif condition == 'extrp':
                masks = cls.extrp[variant]()
            else:
                raise ValueError('Unrecognized condition {}'.format(condition))
        except KeyError:
            raise ValueError('Unrecognized variant {} for condition {}'.format(
                variant, condition))

        if modifiers is not None:
            for mod in modifiers:
                if mod not in cls.modifiers:
                    raise ValueError('Unrecognized modifier {}'.format(mod))

                # If no mask, then modifier is only mask and
                # it is applied during training.
                if masks[0] is None:
                    masks = cls.modifiers[mod], None
                else:
                    modf = partial(compose, mod=cls.modifiers[mod])
                    masks = [(None if m is None else modf(m)) for m in masks]

        return masks


class DummySplits:
    @staticmethod
    def get_splits(condition=None, variant=None, modifiers=None):
        return None, None


## Shapes3D

class _Shapes3D:
    fh, wh, oh, scl, shp, orient = 0, 1, 2, 3, 4, 5

    # Modifies
    @classmethod
    def exclude_odd_ohues(cls, factor_values, factor_classes):
        return (factor_classes[:, cls.oh] % 2) == 0

    @classmethod
    def exclude_half_ohues(cls, factor_values, factor_classes):
        return factor_classes[:, cls.oh] < 5

    @classmethod
    def exclude_odd_wnf_hues(cls, factor_values, factor_classes):
        return (((factor_classes[:, cls.wh] % 2) == 0) &
                ((factor_classes[:, cls.fh] % 2) == 0))

    # Interpolation variants
    @classmethod
    def odd_ohue(cls):
        def train_mask(factor_values, factor_classes):
            return factor_classes[:, cls.oh] % 2 == 0

        def test_mask(factor_values, factor_classes):
            return factor_classes[:, cls.oh] % 2 == 1

        return train_mask, test_mask

    @classmethod
    def odd_wnf_hue(cls):
        def train_mask(factor_values, factor_classes):
            return cls.exclude_odd_wnf_hues(factor_values, factor_classes)

        def test_mask(factor_values, factor_classes):
            return ~cls.exclude_odd_wnf_hues(factor_values, factor_classes)

        return train_mask, test_mask

    # Extrapolation variants
    @classmethod
    def missing_fh_50(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.fh] < 0.5

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    # Recombination to range
    @classmethod
    def ohue_to_whue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oh] >= 0.75) &
                    (factor_values[:, cls.wh] <= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def fhue_to_whue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.fh] >= 0.75) &
                    (factor_values[:, cls.wh] <= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_floor(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.fh] >= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_objh(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] >= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


    @classmethod
    def shape_to_objh_quarter(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] <= 0.25))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_orientation(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:,cls.shp] == 3.0) &
                    (factor_values[:,cls.orient] >= 0))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    # Recombination to element
    @classmethod
    def leave1out(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oh] >= 0.8) &
                    (factor_values[:, cls.wh] >= 0.8) &
                    (factor_values[:, cls.fh] >= 0.8) &
                    (factor_values[:, cls.scl] >= 1.1) &
                    (factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.orient] > 20))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_ohue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:, cls.shp] == 3.0) &
                    (factor_classes[:, cls.oh] == 2))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


class Shapes3D(DataSplit):
    """
    Boolean masks used to partition the Shapes3D dataset
    for each generalisation condition

    #=============================================================
    # Latent Dimension, Latent values
    #=============================================================
    # floor hue:        10 values linearly spaced in [0, 1)
    # wall hue:         10 values linearly spaced in [0, 1)
    # object hue:       10 values linearly spaced in [0, 1)
    # scale:            8 values linearly spaced in [0.75, 1.25]
    # shape:            4 values in [0, 1, 2, 3]
    # orientation:      15 values linearly spaced in [-30, 30]

    """
    interp         = {'odd_ohue'     : _Shapes3D.odd_ohue,
                      'odd_wnf_hue'  : _Shapes3D.odd_wnf_hue}

    recomb2element = {'shape2ohue'   : _Shapes3D.shape_ohue,
                      'leave1out'    : _Shapes3D.leave1out}

    recomb2range   = {'ohue2whue'    : _Shapes3D.ohue_to_whue,
                      'fhue2whue'    : _Shapes3D.fhue_to_whue,
                      'shape2ohue'   : _Shapes3D.shape_to_objh,
                      'shape2ohueq'  : _Shapes3D.shape_to_objh_quarter,
                      'shape2fhue'   : _Shapes3D.shape_to_floor,
                      'shape2orient' : _Shapes3D.shape_to_orientation}

    extrp          = {'missing_fh'   : _Shapes3D.missing_fh_50}

    modifiers      = {'even_ohues'   : _Shapes3D.exclude_odd_ohues,
                      'half_ohues'   : _Shapes3D.exclude_half_ohues,
                      'even_wnf_hues': _Shapes3D.exclude_odd_wnf_hues}
