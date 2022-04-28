import warnings

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, start_value, warmup_steps, warmup_groups=None,
                 last_epoch=-1, verbose=False):
        self.base_lr = np.array([g['lr'] for g in optimizer.param_groups])
        self.start_value = start_value
        self.warmup_steps = warmup_steps

        if warmup_groups is None:
            warmup_groups = np.arange(len(self.base_lr))

        self.warmup_groups = warmup_groups

        super().__init__(optimizer, last_epoch, verbose)

    def step_value(self, i):
        slope = (self.base_lr[i] - self.start_value) / self.warmup_steps
        return self.start_value + self.last_epoch * slope

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            learning_rates = []
            for i, g in enumerate(self.optimizer.param_groups):
                lr = self.step_value(i) if i in self.warmup_groups else g['lr']
                learning_rates.append(lr)
        else:
            learning_rates = [g['lr'] for g in self.optimizer.param_groups]

        return learning_rates


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, decay_groups, last_epoch=-1,
                 verbose=False):
        self.gamma = gamma
        self.decay_groups = decay_groups
        super().__init__(optimizer, last_epoch, verbose)

    def get_gamma(self, i):
        if self.decay_groups is None or i in self.decay_groups:
            return self.gamma
        return 1.0

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.get_gamma(i)
                for i, group in enumerate(self.optimizer.param_groups)]
