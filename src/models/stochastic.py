"""
Module with stochastic layers

Currently only the diagonal gaussian and a couple of variants I was testing
exist here. Original idea was to add more. This just take a mean and a
logvariance and performs the reparameterization trick on the result.

Appart from the standard one, there is a variant with input-independant but
learned logar, wich means that all inputs share the same covariance. This is
similar to the idea behind LDA.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Gaussian stochastic layers
class DiagonalGaussian(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.size = latent_size

    def reparam(self, mu, logvar, random_eval=False):
        if self.training or random_eval:
            # std = exp(log(var))^0.5
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(std)
            # z = mu + std * eps
            return mu.addcmul(std, eps)
        return mu

    def sample(self, inputs, n_samples=1):
        inputs = inputs.unsqueeze_(1).expand(-1, n_samples, -1)
        mu, logvar = inputs.chunk(2, dim=-1)

        return self.reparam(mu, logvar, random_eval=True)

    def forward(self, inputs):
        mu, logvar = inputs.chunk(2, dim=-1)
        return self.reparam(mu, logvar), (mu, logvar)

    def extra_repr(self):
        return 'size={}'.format(self.size)


# Uniform stochastic layers

class Uniform(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.size = latent_size

    def reparam(self, mu, loglen, random_eval=True):
        if self.training or random_eval:
            length = loglen.exp()
            a = mu - length
            b = mu + length
            u = torch.rand_like(a)
            return a.addcmul(u, (b - a))
        return mu

    def sample(self, inputs, n_samples=1):
        inputs = inputs.unsqueeze_(1).expand(-1, n_samples, -1)
        mu, loglen = inputs.chunk(2, dim=-1)

        return self.reparam(mu, loglen, random_eval=True)

    def forward(self, inputs):
        mu, loglen = inputs.chunk(2, dim=-1)
        return self.reparam(mu, loglen, random_eval=True), (mu, loglen)


class BoundedUniform(Uniform):
    def reparam(self, mu, loglen, random_eval=False):
        mu = torch.tanh(mu)
        return super().reparam(mu, loglen, random_eval)


# Discrete variable layer
def cosine_decay(start_value, end_value, n_steps):
    a = 0.5 * (start_value - end_value)
    b = 0.5 * (start_value + end_value)

    def next_step(step):
        if step < n_steps:
            return torch.tensor(a * np.cos(np.pi * step / n_steps) + b)
        return torch.tensor(end_value)

    return next_step


class GumbelSoftmax(nn.Module):
    def __init__(self, input_size, ncat, tau, tau_start=None,
                 tau_steps=None, dim=-1):
        super().__init__()
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
        self.register_buffer('dim', torch.tensor(dim, dtype=torch.int32))
        self.logits = nn.Linear(input_size, ncat)

        if tau_steps is not None:
            self.tau_schedule = cosine_decay(tau_start, tau, tau_steps)
        else:
            self.tau_schedule = None
        self.step = -1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    @property
    def size(self):
        return self.logits.weight.shape[0]

    def forward(self, inputs, hard=None):
        if hard is None:
            hard = not self.training

        if self.training and self.tau_schedule is not None:
            self.step += 1
            self.tau = self.tau_schedule(self.step)

        logits = F.log_softmax(self.logits(inputs), dim=-1)
        z = F.gumbel_softmax(logits, self.tau, hard, dim=-1)

        return z, logits
