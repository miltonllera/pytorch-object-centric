"""
Models used in the experiments, as a Sacred ingredient.

The models just compose different layers from PyTorch or the ``src'' folder
to create the corresponding architecture. The functions use the parsing
functions in ``parsing.py'' to transform a list of layer descriptions into
PyTorch modules. See ``parsing.py'' for more details.
"""


import os
import json
from io import BytesIO

import torch
import torch.nn as nn
# import torch.jit as jit
from sacred import Ingredient

from .parsing import parse_specs
from src.models.autoencoder import AutoEncoder
from src.models.slotattn import SlotAttention, SlotDecoder


model = Ingredient('model')


@model.capture
def init_slot_ae(input_size, n_slots, slot_size, n_iter, hidden_size,
                 encoder_layers, decoder_layers):
    encoder_layers, out_size = parse_specs(input_size, encoder_layers)
    decoder_layers, _ = parse_specs(slot_size, decoder_layers)

    encoder = nn.Sequential(*encoder_layers)
    decoder = SlotDecoder(*decoder_layers)
    slotatt = SlotAttention(out_size, n_slots, slot_size, n_iter, hidden_size)

    # try to optimize with TorchScript
    # example = torch.randn(*input_size).unsqueeze_(0)
    # slotatt = jit.script(slotatt, example_inputs=[example])

    return AutoEncoder(encoder, decoder, slotatt)


@model.capture
def load_slot_ae(input_size, path):
    meta = os.path.join(path, 'config.json')
    param_vals = os.path.join(path, 'trained-model.pt')

    with open(meta) as f:
        architecture = json.load(f)['model']

    slot_ae = init_slot_ae(input_size, **architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    slot_ae.load_state_dict(state_dict)

    return slot_ae.eval()


@model.command(unobserved=True)
def show():
    ae = init_slot_ae((3, 64, 64))
    print(ae)


@model.command(unobserved=True)
def test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ae = init_slot_ae((3, 64, 64)).to(device=device)
    input = torch.randn(10, 3, 64, 64).to(device=device)

    output = ae(input)[0].cpu()

    assert input.shape == output.shape
