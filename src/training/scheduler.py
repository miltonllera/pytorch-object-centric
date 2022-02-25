import warnings

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


class SmoothStepLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_steps, gamma=0.1,
                 last_epoch=-1, verbose=False):
        self.n_steps = n_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed"
                          "by the scheduler, please use `get_last_lr()`.",
                          UserWarning)

        return [g['lr'] * (self.gamma ** (1 / self.n_steps))
                for g in self.optimizer.param_groups]


class WarmupAndDecay(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, decay_steps, warmup_steps,
                 last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.base_lr = np.array([g['lr'] for g in optimizer.param_groups])
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lr = self.base_lr

        if self.last_epoch < self.warmup_steps:
            lr = lr * self.last_epoch / self.warmup_steps

        lr = lr * self.gamma ** (self.last_epoch / self.decay_steps)

        return lr.tolist()
