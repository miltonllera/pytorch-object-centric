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

def discovery():
    """
    Training for object discovery
    """
    n_iters    = 500000
    batch_size = 64
    optimizer  = 'adam'
    lr         = 0.0004
    l2_norm    = 0.0


################################### Predictors #################################

def prediction():
    """
    Training settings for classifiers
    """
    n_iters         = 40000
    batch_size      = 64
    optimizer       = 'adam'
    lr              = 0.001
    train_val_split = 0.2


################################### Schedulers #################################

def exponential_lr():
    scheduler = 'exponential'
    gamma     = 0.5

def step_lr():
    scheduler = 'step'
    gamma     = 0.5
    step_size = 5

def smooth_step_lr():
    scheduler = 'smooth_step'
    gamma     = 0.5
    n_steps   = 100000

def warmup():
    warmup      = True
    start_value = 0.
    duration    = 10000
    end_value   = None  # Will default to optimizer learning rate

def warmup_decay():
    scheduler    = 'warmup_decay'
    gamma        = 0.5
    decay_steps  = 100000
    warmup_steps = 10000
