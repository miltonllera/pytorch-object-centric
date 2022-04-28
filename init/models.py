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
import torch.nn.functional as F
# import torch.jit as jit
from sacred import Ingredient

from .parsing import parse_specs
from src.models.stochastic import GumbelSoftmax
from src.models.autoencoder import AutoEncoder, VariationalAutoEncoder
from src.models.slotattn import SlotAttention, SlotDecoder
from src.models.transformer import TransformerDecoder
from src.models.slate import SLATE
from src.models.initialization import weights_init


model = Ingredient('model')


@model.capture
def init_model(type):
    if type == 'slot_ae':
        return init_slot_ae
    elif type == 'slate':
        return init_slate
    elif type == 'set_predictor':
        return init_set_prediction
    raise ValueError(f"Unrecognized model {type}")


@model.command(unobserved=True)
def show():
    model = init_model()((3, 64, 64))
    print(model)


@model.command(unobserved=True)
def test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = init_model()((3, 64, 64)).to(device=device).train()
    input = torch.randn(10, 3, 64, 64).to(device=device)

    recons, *output = model(input)

    assert input.shape == recons.shape, "Incorrect output shape"

    # Test unsupervised
    loss = ((recons - input) ** 2).sum() / len(input)
    loss.backward()

    if isinstance(model, SLATE):
        tokens, token_logits = output[-1]
        assert token_logits.shape == tokens.shape, "Incorrect token shape"

        # Test slot-transformer
        loss = -(tokens * F.log_softmax(token_logits, dim=-1)).sum()
        loss = loss / len(input)
        loss.backward()

        # Test autoregressive
        model.decode(model.embed(input))


@model.capture
def load_model(input_size, path):
    meta = os.path.join(path, 'config.json')
    param_vals = os.path.join(path, 'trained-model.pt')

    with open(meta) as f:
        architecture = json.load(f)['model']

    model = architecture.pop('type')
    model = init_model(model)(input_size, **architecture)

    with open(param_vals, 'rb') as f:
        state_dict = torch.load(BytesIO(f.read()))

    model.load_state_dict(state_dict)

    return model.eval()


############################## AutoEncoder ####################################

# @model.capture
# def init_ae(input_size, encoder_layers, decoder_layers, vocab_size,
#             tau, tau_start, tau_steps):
#     encoder_layers, out_size = parse_specs(input_size, encoder_layers)
#     latent_output_shape = *out_size[:2], vocab_size
#     decoder_layers, _ = parse_specs(latent_output_shape, decoder_layers)

#     encoder = nn.Sequential(*encoder_layers)
#     decoder = nn.Sequential(*decoder_layers)
#     latent = GumbelSoftmax(out_size[-1], vocab_size,
#                            tau, tau_start, tau_steps)

#     return AutoEncoder(encoder, decoder, latent)


######################### Slot-Attention AutoEncoder ##########################


class SlotAE(AutoEncoder):
    @property
    def slot_attn(self):
        return self.latent


@model.capture
def init_slot_ae(input_size, encoder_layers, decoder_layers, n_slots, slot_size,
                 n_iter, slot_channels, hidden_size, approx_implicit_grad):
    encoder_layers, out_size = parse_specs(input_size, encoder_layers)
    decoder_layers, _ = parse_specs(slot_size, decoder_layers)

    encoder = nn.Sequential(*encoder_layers)
    decoder = SlotDecoder(*decoder_layers)
    slot_attn = SlotAttention(out_size[-1], n_slots, slot_size, n_iter,
                              slot_channels, hidden_size, approx_implicit_grad)

    # try to optimize with TorchScript
    # example = torch.randn(*input_size).unsqueeze_(0)
    # slotatt = jit.script(slotatt, example_inputs=[example])

    return SlotAE(encoder, decoder, slot_attn)



################################# SLATE #######################################


@model.capture
def init_slate(input_size, encoder_layers, decoder_layers, tau, tau_start,
               tau_steps, vocab_size, n_slots, slot_size, n_iter, slot_channels,
               hidden_size, approx_implicit_grad, d_model, nhead, num_layers,
               ffwd_dim, dropout):
    encoder_layers, out_size = parse_specs(input_size, encoder_layers)
    latent_output_shape = *out_size[:2], vocab_size
    decoder_layers, _ = parse_specs(latent_output_shape, decoder_layers)

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)
    latent = GumbelSoftmax(out_size[-1], vocab_size, tau, tau_start, tau_steps)

    dvae = VariationalAutoEncoder(encoder, decoder, latent)
    for m in dvae.children():
        weights_init(m, conv_activation='relu', linear_activation=None)

    seq_len = out_size[0] * out_size[1]
    slot_attn = SlotAttention(d_model, n_slots, slot_size, n_iter,
                              slot_channels, hidden_size, approx_implicit_grad)

    transformer_decoder = TransformerDecoder(seq_len, d_model, nhead,
                                             num_layers, ffwd_dim, dropout)

    return SLATE(out_size[:2], dvae, slot_attn, transformer_decoder, True)


########################## Property Prediction #################################


class EmbeddingModel(nn.Module):
    def __init__(self, model, output_idx=None):
        super().__init__()
        self.model = model
        self.output_dim = output_idx

    def forward(self, inputs):
        out = self.model.embed(inputs)

        if self.output_dim is not None:
            out = out[self.output_dim]

        return out


@model.capture
def init_set_prediction(input_size, n_targets, slot_model_path,
                        layer_size=64, n_layers=1, activation='relu'):
    # Slot-based front end
    slot_model = load_model(input_size, slot_model_path)
    for p in slot_model.parameters():
        p.requires_grad = False

    if isinstance(slot_model, SLATE):
        input_size = slot_model.slot_attn.slot_size
    else:
        input_size = slot_model.latent.slot_size

    mlp_layers = [EmbeddingModel(slot_model, output_idx=0)]
    for i in range(n_layers):
        mlp_layers.extend([nn.Linear(input_size, layer_size), nn.ReLU()])
        input_size = layer_size

    mlp_layers.extend([nn.Linear(input_size, n_targets), nn.Sigmoid()])

    return nn.Sequential(*mlp_layers)
