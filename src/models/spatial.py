from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from .initialization import linear_init


def cardinal_pos_embedding(resolution, min_val=0, max_val=1.0):
    grid = origin_pos_embedding(resolution, min_val, max_val)
    return torch.cat([grid, 1.0 - grid], dim=0)


def origin_pos_embedding(resolution, min_val=-1.0, max_val=1.0):
    ranges = [torch.linspace(min_val, max_val, steps=r) for r in resolution]
    grid = torch.meshgrid(*ranges, indexing='xy')
    grid = torch.stack(grid, dim=0)
    return grid.to(dtype=torch.float32)


class PositionConcat(nn.Module):
    def __init__(self, height, width=None, dim=-3, embed='origin'):
        super().__init__()

        if width is None:
            width = height

        self.height = height
        self.width = width

        if embed == 'cardinal':
            grid = cardinal_pos_embedding((height, width))
        elif embed == 'origin':
            grid = origin_pos_embedding((height, width))
        else:
            raise ValueError('Unrecognized embedding type {}'.format(embed))

        self.grid = grid
        self.dim = dim

    def forward(self, inputs):
        sizes = list(inputs.shape()[:-3]) + [-1, -1, -1]
        grid = self.grid.expand(sizes).to(device=inputs.device)
        return torch.cat([inputs, grid], dim=self.dim).contiguous()

    def __repr__(self):
        return 'PositionConcat(height={},width={})'.format(self.height,
                                                           self.width)


class PositionEmbedding2D(nn.Module):
    def __init__(self, n_channels, height, width=None, embed='cardinal'):
        super().__init__()

        if width is None:
            width = height

        self.height = height
        self.width = width

        if embed == 'cardinal':
            grid = cardinal_pos_embedding((height, width))
            linear = nn.Linear(4, n_channels)
        elif embed == 'origin':
            grid = origin_pos_embedding((height, width))
            linear = nn.Linear(2, n_channels)
        else:
            raise ValueError('Unrecognized embedding type {}'.format(embed))

        linear_init(linear, activation=None)

        self.grid = grid.T
        self.projection = linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        proj = self.projection(self.grid.to(device=inputs.device))
        return inputs + proj

    def reset_parameters(self):
        linear_init(self.projection, activation=None)

    def __repr__(self):
        return 'PositionEmbedding2D(height={}, width={})'.format(
                self.height, self.width)


class SpatialBroadcast(nn.Module):
    def __init__(self, height, width=None, input_last=False) -> None:
        super().__init__()

        if width is  None:
            width = height

        self.width = width
        self.height = height
        self.input_last = input_last

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bdc = torch.tile(inputs[(..., None, None)], (self.height, self.width))
        if self.input_last:
            bdc = torch.movedim(bdc, -3, -1)
        return bdc

    def __repr__(self):
        return 'SpatialBroadcast(height={},width={}, input_last={})'.format(
                self.height, self.width, self.input_last)


# class WeightedSBC(SpatialBroadcast):
#     def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
#         z, mask = inputs

#         # tiled, shape (bs * n_slots, slot_size, height, width)
#         tiled = torch.tile(z[(..., None, None)], (self.height, self.width))
#         mask = mask.unsqueeze(1).unflatten(-1, (self.height, self.width))

#         return tiled * mask
