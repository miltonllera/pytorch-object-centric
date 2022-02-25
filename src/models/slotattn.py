import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Tuple, Union, get_type_hints
from .initialization import linear_init, gru_init


class SlotAttention(nn.Module):
    def __init__(self, input_size: tuple, n_slots: int = 4, slot_size: int = 64,
            n_iter: int = 3, hidden_size: int = 128, epsilon: float = 1e-8):
        super().__init__()

        assert n_slots > 1, "Must have at least two slots"
        assert n_iter > 0, "Need at least one slot update iteration"

        _, input_size = input_size

        self.n_slots = n_slots
        self.slot_size = slot_size
        self.n_iter = n_iter
        self.EPS = epsilon

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
        gru_init(self.gru, bias=True)

        for m in self.children():
            try:
                m.reset_parameters()
            except AttributeError:
                pass

    def forward(self, inputs: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        slots = self.init_slots(inputs)  # batch_size, n_slots, slot_size
        inputs = self.norm_input(inputs)  # batch_size, n_inputs, input_size

        # keys and values, shape (batch_size, n_inputs, slot_size)
        k = self.k_proj(inputs) / (self.slot_size ** 0.5)
        v = self.v_proj(inputs)

        for _ in range(self.n_iter):
            slots, atten_weights = self.step(slots, k, v)

        # First-order Neumann approximation to implicit gradient (Cheng et al)
        slots, atten_weights = self.step(slots.detach(), k, v)

        return slots, atten_weights

    def init_slots(self, inputs):
        std = self.slot_logvar.mul(0.5).exp()
        std = std.expand(len(inputs), self.n_slots, -1)
        eps = torch.randn_like(std)
        return self.slot_mu.addcmul(std, eps).to(device=inputs.device)

    def step(self, slots, k, v):
        norm_slots = self.norm_slot(slots)
        q = self.q_proj(norm_slots)

        # both of shape (batch_sizs, n_slots, slot_size)
        atten_maps, atten_weights = self.compute_attention_maps(k, q, v)

        slots = self.update_slots(atten_maps, slots)

        return slots, atten_weights

    def compute_attention_maps(self, k, q, v):
        # slots compete for input patches
        weights = k.matmul(q.transpose(1, 2))  # n_inputs, n_slots
        weights = F.softmax(weights, dim=-1)

        # weighted mean over n_inputs
        weights = weights + self.EPS
        weights = weights / weights.sum(dim=-2, keepdim=True)

        atten_maps = weights.transpose(1, 2).matmul(v)

        return atten_maps, weights

    def update_slots(self, atten_maps, slots):
        B = len(slots)

        # batchify update
        atten_maps = atten_maps.flatten(end_dim=1)
        slots = slots.flatten(end_dim=1)

        # update slots
        slots = self.gru(atten_maps, slots)
        slots = slots + self.mlp(self.norm_res(slots))

        slots = slots.unflatten(0, (B, self.n_slots))

        return slots

    def __repr__(self):
        return 'SlotAttention(n_slots={}, slot_size={}, n_iter={})'.format(
            self.n_slots, self.slot_size, self.n_iter)


class SlotDecoder(nn.Sequential):
    def _pass_slot_only(self):
        types = get_type_hints(self[0].forward)
        return types['inputs'] == torch.Tensor

    def decode_slots(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
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

    def forward(self, inputs: Tensor) -> Tensor:
        slot_recons, slot_masks = self.decode_slots(inputs)
        return (slot_masks * slot_recons).sum(dim=1)

    def masks(self, inputs: Tensor) -> Tensor:
        return self.decode_slots(inputs)[1]
