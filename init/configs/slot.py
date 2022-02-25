"""
Autoencoder model definitions

Following the Sacred configuration model using python functions, we define the
architectures of the models here. The names referece the first author of the
article from where they were take. Some might be slightly modified.

Configurations follow a general structure:
    1. gm_type (currently only lgm is available)
    2. latent_size
    3. input_size (not neccessary since it is overwritten depending on the dataset)
    4. encoder_layers: a list with layer definitions
    5. decoder layers: optional, model creation function will attemtp to transpose

Parameters in the config for each layer follow the order in Pytorch's documentation
Excluding any of them will use the default ones. We can also pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                  | Paramaeters
================================================================================
Convolution            : n-channels, size, stride, padding
Transposed Convolution : same, remeber output_padding when stride>1 (use kwargs)
Pooling                : size, stride, padding, type
Linear                 : output size, fit bias
batchnorm              : dimensionality (1-2-3d)
LayerNorm              : NA
SoftPositionalEmbedding: NA
PositionConcat         : NA
SpatialBroadcast       : height, width (optional, defaults=height)
Flatten                : dim, (optional, defaults=-1) end dim
Unflatten              : unflatten shape (have to pass the full shape)
Upsample               : upsample_shape (hard to infer automatically).
Non-linearity          : pass whatever arguments that non-linearity supports.
"""

def slotattn():
    # SlotAttention params
    n_slots = 4
    slot_size = 64
    n_iter = 3
    hidden_size = 128


def small_ae():
    # Encoder
    encoder_layers = [
        # CNN block
        ('conv', (32, 5, 1, 'same')),
        ('relu',),
        ('conv', (32, 5, 1, 'same')),
        ('relu',),
        ('conv', (32, 5, 1, 'same')),
        ('relu',),
        ('conv', (32, 5, 1, 'same')),
        ('relu',),

        # Add position info
        ('permute', (0, 2, 3, 1)),
        ('posemb2d', {'embed': 'cardinal'}),

        # Per position MLP
        ('flatten', [1, 2]),
        ('layer_norm', [-1]),

        ('linear', [32]),
        ('relu',),
        ('linear', [32]),
    ]

    # Decoder
    decoder_layers = [
        # Spatial Broadcast
        ('spatbroad', (35, 35), {'input_last': True}),
        ('posemb2d', {'embed': 'cardinal'}),
        ('permute', (0, 3, 2, 1)),

        # De-Convolution block
        ('tconv', (32, 5, 1, 2)),  # Padding so output is equivalent to "SAME"
        ('relu',),
        ('tconv', (32, 5, 1, 2)),
        ('relu',),
        ('tconv', (32, 5, 1, 2)),
        ('relu',),

        # Decoded masks and perslot images
        ('tconv', (4, 3, 1, 1)),
    ]


def medium_ae():
    encoder_layers = [
        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('permute', (0, 2, 3, 1)),

        # Add position info
        ('posemb2d', {'embed': 'cardinal'}),

        # Per position MLP
        ('flatten', [1, 2]),
        ('layer_norm', [-1]),

        ('linear', [64]),
        ('relu',),
        ('linear', [64]),
    ]

    decoder_layers = [
        ('spatbroad', (8, 8), {'input_last': True}),
        ('posemb2d', {'embed': 'cardinal'}),

        ('permute', (0, 3, 2, 1)),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (4, 4, 2, 1)),
    ]


def large_ae():
    # Encoder
    encoder_layers = [
        # CNN block
        ('conv', (64, 5, 1, 'same')),
        ('relu',),
        ('conv', (64, 5, 1, 'same')),
        ('relu',),
        ('conv', (64, 5, 1, 'same')),
        ('relu',),
        ('conv', (64, 5, 1, 'same')),
        ('relu',),

        # Add position info
        ('permute', (0, 2, 3, 1)),
        ('posemb2d', {'embed': 'cardinal'}),

        # Per position MLP
        ('flatten', [1, 2]),
        ('layer_norm', [-1]),

        ('linear', [64]),
        ('relu',),
        ('linear', [64]),
    ]

    # Decoder
    decoder_layers = [
        # Spatial Broadcast
        ('spatbroad', (8, 8), {'input_last': True}),
        ('posemb2d', {'embed': 'cardinal'}),
        ('permute', (0, 3, 2, 1)),

        # Convolution block
        ('tconv', (64, 4, 2, 1)),
        ('relu',),
        ('tconv', (64, 4, 2, 1)),
        ('relu',),
        ('tconv', (64, 4, 2, 1)),
        ('relu',),
        ('tconv', (64, 4, 2, 1)),
        ('relu',),
        ('tconv', (64, 4, 1, 1)),
        ('relu',),

        # Decoded masks and perslot images
        ('tconv', (4, 4, 1, 1)),
    ]


# def soft_sbd():
#     decoder_layers = [
#         # Spatial Broadcast
#         ('spatbroad', (64, 64), {'type': 'weighted'}),
#         ('softpemb', [-3], {'channels_last': False}),

#         # Convolution block
#         ('conv', (64, 5, 1, 'same')),
#         ('relu',),
#         ('conv', (64, 5, 1, 'same')),
#         ('relu',),
#         ('conv', (64, 5, 1, 'same')),
#         ('relu',),
#         ('conv', (64, 5, 1, 'same')),
#         ('relu',),
#         ('conv', (64, 5, 1, 'same')),
#         ('relu',),

#         # Decoded masks and perslot images
#         ('conv', (4, 5, 1, 'same')),
#     ]
