import torch
import torch.nn as nn
from .initialization import linear_init


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, max_seqlen, d_model=512, nhead=4, num_layers=4,
                 ffwd_dim=2048, dropout=0.0):

        layer_norm = nn.LayerNorm(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, ffwd_dim,
                dropout, batch_first=True, norm_first=True)

        super().__init__(decoder_layer, num_layers, layer_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.mask = get_autoregressive_mask(max_seqlen)

        init_projections(self, num_layers)

    def forward(self, tgt, memory, memory_mask=None):
        L = tgt.shape[1]  # required for autoregressive generation

        mask = self.mask[:L, :L].to(device=tgt.device)
        if memory_mask is not None:
            # memory_mask must match batch_size * nhead for MultiheadAttention
            memory_mask = memory_mask[None].expand(self.nhead, -1, -1, -1)
            memory_mask = memory_mask.flatten(end_dim=1)[:, :L]

        return super().forward(tgt, memory, mask, memory_mask)


def get_autoregressive_mask(size):
    return torch.ones((size, size)).triu(diagonal=1).to(dtype=torch.bool)


def init_projections(transformer, num_layers):
    gain = (3 * num_layers) ** (-0.5)
    for mod in transformer.children():
        if isinstance(mod, nn.MultiheadAttention):
            linear_init(mod.out_proj, activation=None, gain=gain)
        if isinstance(mod, nn.TransformerDecoderLayer):
            linear_init(mod.linear2, activation=None, gain=gain)
