from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss
from torch.nn.modules.loss import _Loss


# Metrics
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
