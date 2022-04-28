import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Tuple, get_type_hints
from .initialization import linear_init, gru_init
from .autoencoder import AutoEncoder


class SlotAttention(nn.Module):
    def __init__(self, input_size: int, n_slots: int = 4, slot_size: int = 64,
            n_iter: int = 3, slot_channels=1, hidden_size: int = 128,
            approx_implicit_grad: bool = True, epsilon: float = 1e-8):
        super().__init__()

        assert n_slots > 1, "Must have at least two slots"
        assert n_iter > 0, "Need at least one slot update iteration"
        assert (slot_size % slot_channels) == 0

        self.n_slots = n_slots
        self.slot_size = slot_size
        self.n_iter = n_iter
        self.nhead = slot_channels
        self.EPS = epsilon
        self.approx_implicit_grad = approx_implicit_grad

        # slot init
        self.slot_mu = Parameter(torch.empty(1, 1, slot_size))
        self.slot_logvar = Parameter(torch.empty(1, 1, slot_size))

        # attention projections
        self.k_proj = nn.Linear(input_size, slot_size, bias=False)
        self.v_proj = nn.Linear(input_size, slot_size, bias=False)
        self.q_proj = nn.Linear(slot_size, slot_size, bias=False)

        # normalisation layers
        self.norm_input = nn.LayerNorm(input_size)
        self.norm_slot = nn.LayerNorm(slot_size)
        self.norm_res = nn.LayerNorm(slot_size)

        # update
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(nn.Linear(slot_size, hidden_size, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, slot_size))

        self.reset_parameters()

    @property
    def size(self):
        return self.n_slots * self.slot_size

    @property
    def hidden_size(self):
        return self.res_mlp[0].out_features

    @property
    def shape(self):
        return self.n_slots, self.slot_size

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_logvar)

        linear_init(self.k_proj, activation='relu')
        linear_init(self.v_proj, activation='relu')
        linear_init(self.q_proj, activation='relu')

        for m in self.mlp.children():
            if isinstance(m, nn.Linear):
                linear_init(m, activation='relu')

        gru_init(self.gru)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        slots = self.init_slots(inputs)  # batch_size, n_slots, slot_size
        inputs = self.norm_input(inputs)  # batch_size, n_inputs, input_size

        # shape (batch_size, n_heads, n_inputs, slot_size // nheads)
        k = self.split_heads(self.k_proj(inputs))
        v = self.split_heads(self.v_proj(inputs))

        k = k / ((self.slot_size / self.nhead) ** 0.5)

        for _ in range(self.n_iter):
            slots, atten_masks = self.step(slots, k, v)

        # First-order Neumann approximation to implicit gradient (Chang et al)
        if self.approx_implicit_grad:
            slots, atten_masks = self.step(slots.detach(), k, v)

        # slots are in the correct shape
        return slots, atten_masks.sum(dim=1)  # add weights from attention heads

    def split_heads(self, input):
        # input size: B, n_in, slot_size/input_size
        split_size = input.shape[-1] // self.nhead
        return input.unflatten(-1, (self.nhead, split_size)).transpose(1, 2)

    def join_heads(self, input):
        # input size: B, n_head, n_in, slot_size/input_size // n_head
        return input.transpose(1, 2).flatten(start_dim=2)

    def init_slots(self, inputs):
        std = self.slot_logvar.mul(0.5).exp()
        std = std.expand(len(inputs), self.n_slots, -1)
        eps = torch.randn_like(std)
        return self.slot_mu.addcmul(std, eps)

    def step(self, slots, k, v):
        q = self.q_proj(self.norm_slot(slots))

        # atten_maps: (batch_sizs, n_slots, slot_size)
        # atten_weights: (batch_size, n_heads, n_slots, slot_size // n_heads)
        atten_maps, atten_weights = self.compute_attention_maps(k, q, v)

        slots = self.update_slots(atten_maps, slots)

        return slots, atten_weights

    def compute_attention_maps(self, k, q, v):
        q = self.split_heads(q)

        # slots compete for input patches
        weights = k.matmul(q.transpose(2, 3))  # n_inputs, n_slots
        weights = F.softmax(self.join_heads(weights), dim=-1)

        # weighted mean over n_inputs
        weights = self.split_heads(weights) + self.EPS
        weights = weights / weights.sum(dim=-2, keepdim=True)

        atten_maps = self.join_heads(weights.transpose(2, 3).matmul(v))

        return atten_maps, weights

    def update_slots(self, atten_maps, slots):
        B = len(slots)

        # batchify update
        atten_maps = atten_maps.flatten(end_dim=1)
        slots = slots.flatten(end_dim=1)

        # update slots
        slots = self.gru(atten_maps, slots)
        slots = slots + self.mlp(self.norm_res(slots))

        return slots.unflatten(0, (B, self.n_slots))

    def __repr__(self):
        return 'SlotAttention(n_slots={}, slot_size={}, n_iter={})'.format(
            self.n_slots, self.slot_size, self.n_iter)


class SlotDecoder(nn.Sequential):
    def _pass_slot_only(self):
        types = get_type_hints(self[0].forward)
        return types['inputs'] == torch.Tensor

    def decode_slots(self, inputs):
        slots, atten_weights = inputs
        batch_size, n_slots = slots.size()[:2]

        # batchify reconstruction
        slots = slots.flatten(end_dim=1)
        atten_weights = atten_weights.flatten(end_dim=1)

        if self._pass_slot_only():  # pass attention weights to decoder
            rgba = super().forward(slots)
        else:
            rgba = super().forward((slots, atten_weights))

        #  shape: (batch_size, n_slots, n_channels + 1, height, width)
        rgba = rgba.unflatten(0, (batch_size, n_slots))

        slot_recons, slot_masks = torch.tensor_split(rgba, indices=[3], dim=2)
        slot_masks = F.softmax(slot_masks, dim=1)  # masks are logits

        return slot_recons, slot_masks

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        slot_recons, slot_masks = self.decode_slots(inputs)
        return (slot_masks * slot_recons).sum(dim=1)
