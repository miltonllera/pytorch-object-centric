"""
Initialization of weigths for autoencoders.

Taken from:
    https://github.com/YannDubs/disentangling-vae/master/disvae/utils/initialization.py
"""


import torch
import torch.nn as nn
import torch.nn.init as init


def get_activation_name(activation):
    """
    Given a string or a `torch.nn.modules.activation`
    return the name of the activation.
    """
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=init.calculate_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)


def xavier_normal_init_(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass


def kaiming_normal_init_(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        try:
            m.bias.data.zero_()
        except AttributeError:
            pass


def gru_init(gru, bias=True):
    nn.init.xavier_uniform_(gru.weight_ih)
    nn.init.orthogonal_(gru.weight_hh)

    if bias:
        nn.init.zeros_(gru.bias_ih)
        nn.init.zeros_(gru.bias_hh)
