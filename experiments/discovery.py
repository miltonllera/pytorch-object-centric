"""
To run:
    cd <experiment-root>/
    python -m experiment.discovery with dataset.<option> \
                                        model.<option> training.<option>

Additional configuration options can be achieved as explained in the Sacred documentation
[https://sacred.readthedocs.io/en/stable/]

"""


import numpy as np
import torch

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer
from ignite.contrib.handlers import ProgressBar

# Load experiment ingredients and their respective configs.
from init.dataset import dataset, get_dataset, get_data_spliters
from init.models import model, init_slot_ae
from init.training import training, init_loader, get_n_epochs, \
                          ModelCheckpoint, init_loss, init_metrics, \
                          init_optimizer, init_lr_scheduler, mse_recons

from init.configs import training as train_params
from init.configs import slot as model_params


ex = Experiment(name='discovery', ingredients=[dataset, model, training])

# Required for ProgressBar to work properly
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

ex.observers.append(FileStorageObserver.create('data/sims/discovery'))

# General configs
ex.add_config(no_cuda=False, temp_folder='data/sims/temp')
ex.add_package_dependency('torch', torch.__version__)


# Functions
@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


# Dataset configs
dataset.add_config(setting='unsupervised', shuffle=True)

# Training configs
training.add_config(loss=mse_recons, metrics=[mse_recons], scheduler=None)
training.config(train_params.discovery)

training.named_config(train_params.exponential_lr)
training.named_config(train_params.step_lr)
training.named_config(train_params.smooth_step_lr)
training.named_config(train_params.warmup)
training.named_config(train_params.warmup_decay)

# Model configs
model.config(model_params.slotattn)
model.config(model_params.small_ae)
model.named_config(model_params.large_ae)
model.named_config(model_params.medium_ae)
model.named_config(model_params.small_ae)
# model.named_config(model_params.soft_sbd)


# Run experiment
@ex.automain
def main(_config):
    device = set_seed_and_device()
    # Load data
    data_filters = get_data_spliters()
    dataset = get_dataset(data_filters=data_filters, train=True)

    training_loader, validation_loader = init_loader(dataset)

    # Init model
    img_size = training_loader.dataset.img_size
    model = init_slot_ae(input_size=img_size).to(device=device)

    # Init metrics
    loss, metrics = init_loss(), init_metrics()

    # Init optimizer
    optimizer = init_optimizer(params=model.parameters())
    scheduler = init_lr_scheduler(optimizer)

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    ProgressBar().attach(trainer,
            output_transform= lambda x: {'loss': f"{x: >10.2f}"})

    validator = create_supervised_evaluator(model, metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_loader,
                      epoch_length=np.ceil(len(validation_loader) * 0.2))

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.terminate()

    # Update lr
    if scheduler is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

    # Record training progression
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        loss = engine.state.output
        ex.log_scalar('training_loss', loss)

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    best_checkpoint = ModelCheckpoint(
        dirname=_config['temp_folder'],
        filename_prefix='discovery_best',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(Events.COMPLETED, best_checkpoint,
                                {'model': model})

    # Run the training
    trainer.run(training_loader, max_epochs=get_n_epochs(training_loader))
    # Select best model
    model.load_state_dict(best_checkpoint.last_checkpoint_state)

    # # Save best model performance and state
    ex.add_artifact(best_checkpoint.last_checkpoint, 'trained-model.pt')
