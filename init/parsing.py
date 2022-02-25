"""
Layer parsing

These functions parse a list of layer configurations into a list of PyTorch
modules. Parameters in the config for each layer follow the order in Pytorch's
documentation. Excluding any of them will use the default ones. We can also
pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                   Paramaeters
==============================================================================
Convolution           : n-channels, size, stride, padding
Transposed Convolution: same, output_padding when stride > 1! (use kwargs)
Pooling               : size, stride, padding, type
Linear                : output size, fit bias
Flatten               : start dim, (optional, defaults=-1) end dim
Unflatten             : unflatten shape (have to pass the full shape)
Batch-norm            : dimensionality (1-2-3d)
Upsample              : upsample_shape (hard to infer automatically). Bilinear
Non-linearity         : pass whatever arguments that non-linearity supports.
SpatialBroadcast      : height, width (optional, defaults to value of height)

There is a method called transpose_layer_defs which allows for automatically
transposing the layer definitions for a decoder in a generative model. This
will automatically convert convolutions into transposed convolutions and
flattening to unflattening. However it will produce weird (but functionally
equivalent) orders of layers for ReLU before flattening, which means
unflattening in the corresponding decoder will be done before the ReLU.
"""


import sys
import numpy as np
import torch
import torch.nn as nn
from .math import *

from src.models.spatial import PositionConcat, PositionEmbedding2D, \
                               SpatialBroadcast


def preprocess_defs(layer_defs):
    def preprocess(definition):
        if len(definition) == 1:
            return definition[0], [], {}
        elif len(definition) == 2 and isinstance(definition[1], (tuple, list)):
            return (*definition, {})
        elif len(definition) == 2 and isinstance(definition[1], dict):
            return definition[0], [], definition[1]
        elif len(definition) == 3:
            return definition
        raise ValueError('Invalid layer definition')

    return list(map(preprocess, layer_defs))


def get_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return nn.ReLU
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid
    elif nonlinearity == 'tanh':
        return nn.Tanh
    elif nonlinearity == 'lrelu':
        return nn.LeakyReLU
    elif nonlinearity == 'elu':
        return nn.ELU
    raise ValueError('Unrecognized non linearity: {}'.format(nonlinearity))


def create_linear(input_size, args, kwargs, transposed=False):
    if isinstance(input_size, tuple):
        input_size = list(input_size)

    if isinstance(input_size, list):
        in_features = input_size[-1]
    else:
        in_features = input_size

    if transposed:
        layer = nn.Linear(args[0], in_features, *args[1:], **kwargs)
    else:
        layer = nn.Linear(in_features, *args, **kwargs)

    if isinstance(input_size, list):
        input_size[-1] = args[0]
    else:
        input_size = args[0]

    return layer, input_size


def create_pool(input_size, args, kwargs):
    kernel_size, stride, padding, mode = args
    if mode == 'avg':
        layer = nn.AvgPool2d(kernel_size, stride, padding, **kwargs)
        #TODO: implement layer output shape computation
    elif mode == 'max':
        layer = nn.MaxPool2d(kernel_size, stride, **kwargs)
        output_size = maxpool2d_out_shape(input_size, kernel_size,
                                          stride, padding)
    elif mode == 'adapt':
        layer = nn.AdaptiveAvgPool2d(kernel_size, **kwargs)
        #TODO: implement layer output shape computation
    else:
        raise ValueError('Unrecognised pooling mode {}'.format(mode))
    return pooling, output_size


def create_batch_norm(ndims, input_size, args, kwargs):
    if ndims == 1:
        return nn.BatchNorm1d(input_size, *args, **kwargs)
    if ndims == 2:
        return nn.BatchNorm2d(input_size[0], *args, **kwargs)
    if ndims == 3:
        return nn.BatchNorm3d(input_size[0], *args, **kwargs)
    raise ValueError('Unknown number of dimension for Batch-norm')


def create_layer_norm(input_size, args, kwargs):
    if isinstance(input_size, tuple):
        normalized_shape = input_size[args[0]:]
    else:
        normalized_shape = input_size
    return nn.LayerNorm(normalized_shape, *args[1:], **kwargs)


def create_conv(input_size, args, kwargs):
    layer = nn.Conv2d(input_size[0], *args, **kwargs)
    output_size = conv2d_out_shape(input_size, *args)
    return layer, output_size


def create_tconv(input_size, args, kwargs):
    layer = nn.ConvTranspose2d(input_size[0], *args, **kwargs)
    output_size = transp_conv2d_out_shape(input_size, *args)
    return layer, output_size


def create_spatbroad(input_size, args, kwargs):
    kwargs = kwargs.copy()

    input_last = kwargs['input_last']
    sbc_type = kwargs.pop('type', 'uniform')

    if sbc_type == 'uniform':
        layer = SpatialBroadcast(*args, **kwargs)
    # elif sbc_type == 'weighted':
    #     layer = WeightedSBC(*args, **kwargs)
    else:
        raise ValueError('Unrecognized spatial'
                         'broadcast type {}'.format(sbc_type))

    if input_last:
        output_size = (*args, input_size)
    else:
        output_size = (input_size, *args)

    return layer, output_size


def create_posconcat(input_size, args, kwargs):
    layer = PositionConcat(*args, **kwargs)
    output_size = [layer.height, layer.width]
    output_size.insert(layer.dim, input_size + 2)
    return layer, output_size


def create_posemb(input_size, args, kwargs):
    height, width, dim_size = input_size
    return PositionEmbedding2D(dim_size, height, width, **kwargs)


class Permute(nn.Module):
    def __init__(self, new_order) -> None:
        super().__init__()
        self.new_order = new_order

    def forward(self, inputs):
        return inputs.permute(self.new_order)

    def __repr__(self):
        return 'Permute({})'.format(*self.new_order)


def create_permute(input_size, args, kwargs):
    layer = Permute(args)
    output_size = list(np.array(input_size)[np.asarray(args[1:]) - 1])
    return layer, output_size


def parse_specs(input_size, layer_defs):
    if isinstance(layer_defs, dict):
        layer_defs = layer_defs.items()
    layer_defs = preprocess_defs(layer_defs)

    module_layers, output_size = [], input_size

    for layer_type, args, kwargs in layer_defs:
        if layer_type == 'linear':
            layer, output_size = create_linear(output_size, args, kwargs)
        elif layer_type == 'conv':
            layer, output_size = create_conv(output_size, args, kwargs)
        elif layer_type == 'tconv':
            layer, output_size = create_tconv(output_size, args, kwargs)
        elif layer_type == 'batch_norm':
            layer = create_batch_norm(args[0], output_size, args[1:], kwargs)
        elif layer_type == 'layer_norm':
            layer = create_layer_norm(output_size, args, kwargs)
        elif layer_type == 'pool':
            layer, output_size = create_pool(output_size, args, kwargs)
        elif layer_type == 'dropout':
            layer = nn.Dropout2d(*args, **kwargs)
        elif layer_type == 'flatten':
            layer = nn.Flatten(*args)
            output_size = compute_flattened_size(output_size,  *args)
        elif layer_type == 'unflatten':
            layer = nn.Unflatten(*args)
            output_size = args[1]
        elif layer_type == 'upsample':
            layer = nn.UpsamplingBilinear2d(*args)
            output_size = output_size[0], *args[0]
        elif layer_type == 'spatbroad':
            layer, output_size = create_spatbroad(output_size, args, kwargs)
        elif layer_type == 'posemb2d':
            layer = create_posemb(output_size, args, kwargs)
        elif layer_type == 'posconcat':
            layer, output_size = create_posconcat(output_size, args, kwargs)
        elif layer_type == 'permute':
            layer, output_size = create_permute(output_size, args, kwargs)
        else:
            layer = get_nonlinearity(layer_type)(*args, **kwargs)

        module_layers.append(layer)

    return module_layers, output_size
