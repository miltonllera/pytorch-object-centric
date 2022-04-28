"""
Following the Sacred configuration model using python functions, we define the
architectures of the models here. The names reference the loss functions as
defined in the corresponding articles (see src/training/loss.py). Both use the
following format:

    {'name': <name>, 'params': <dict_of_kwargs>, 'label': <alt_name>}

The parameters are the ones specified by each fuction (check the file above).
Alt name is used to resolve conflcits between metrics with the same parameters^*.

"""


################################ Reconstruction ################################

def slot_ae():
    """
    Training for object discovery
    """
    n_iters    = 500000
    batch_size = 64
    optimizer  = 'adam'
    lr         = 0.0004


def dvae():
    """
    Training for dVAE
    """
    n_iters    = 300000
    batch_size = 50
    optimizer  = 'adam'
    lr         = 0.0003


def slate():
    """
    Training for slate
    """
    n_iters      = 500000
    batch_size   = 50
    optimizer    = 'adam'
    lr           = 0.0001
    # Use same names as in SLATE module for the parameter groups
    param_groups = [('patch_ae', {'lr': 0.0003})]


################################### Predictors #################################

def prediction():
    """
    Training settings for property prediction
    """
    n_iters         = 40000
    batch_size      = 64
    optimizer       = 'adam'
    lr              = 0.001
    train_val_split = 0.2


################################### Schedulers #################################

def reduce_lr_on_plateau():
    reduce_on_plateau = True
    factor    = 0.5
    patience  = 4

def exponential_lr():
    scheduler    = 'exponential'
    gamma        = 0.5 ** (1/100000)  # number of steps until lr is halved
    decay_groups = [1]

def warmup():
    warmup        = True
    start_value   = 0.0
    warmup_steps  = 30000
    warmup_groups = [1]  # warmup main params (slot, gpt-decoder & embedding)
