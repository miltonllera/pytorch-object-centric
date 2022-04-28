import torch
import torch.multiprocessing as mp
from torch.nn.functional import mse_loss, log_softmax, huber_loss
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.modules.loss import _Loss
from scipy.optimize import linear_sum_assignment as lsa


class ReconstructionNLL(_Loss):
    """
    Standard reconstruction of images. There are two options, minimize the
    per-pixel binary cross entropy (Bernoulli loss) or the per-pixel
    mean squared error (MSE).
    """
    def __init__(self, loss='bce'):
        super().__init__(reduction='batchmean')
        if loss == 'bce':
            recons_loss = logits_bce
        elif loss == 'mse':
            recons_loss = mse_loss
        elif not callable(loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(loss))
        self.loss = recons_loss

    def forward(self, input, target):
        if isinstance(input, (tuple, list)):
            recons = input[0]
        else:
            recons = input

        return self.loss(recons, target, reduction='sum') / target.size(0)


class ImageTokenLoss(_Loss):
    def __init__(self):
        super().__init__(reduction='batchmean')

    def forward(self, inputs, targets):
        return -(targets * log_softmax(inputs, dim=-1)).sum() / len(inputs)


class SLATELoss(_Loss):
    def __init__(self, loss='mse'):
        super().__init__(reduction='batchmean')
        self.recons_loss = ReconstructionNLL(loss=loss)
        self.st_loss = ImageTokenLoss()

    def forward(self, inputs, targets):
        recons, _, _, transformer_output = inputs

        recons_mse = self.recons_loss(recons, targets)
        st_cross_entropy = self.st_loss(*transformer_output)

        return recons_mse + st_cross_entropy


def huber_norm(x, y):
    return huber_loss(x, y, reduction='none').sum(-1)


def l2_norm(x, y):
    return ((x - y) ** 2).sum(-1)


def l1_norm(x, y):
    return (x - y).abs().sum(-1)


class HungarianLoss(_Loss):
    def __init__(self, loss='huber'):
        super().__init__(reduction='batchmean')
        if loss == 'huber':
            loss = huber_norm
        elif loss == 'l2':
            loss = l2_norm
        elif loss == 'l1':
            loss = l1_norm
        self.loss = loss

    def forward(self, inputs, targets):
        B = len(inputs)

        targets = targets.unsqueeze(1).expand(-1, inputs.size(1), -1, -1)
        inputs = inputs.unsqueeze(2).expand(-1, -1, targets.size(2), -1)

        pairwise_cost = self.loss(inputs, targets)

        with mp.Pool(10) as p:
            idx_input, idx_targets = list(zip(p.map(lsa, pairwise_cost)))

        return pairwise_cost[torch.arange(B), idx_input, idx_targets].sum() / B
