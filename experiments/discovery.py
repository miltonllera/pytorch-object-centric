from functools import partial
import numpy as np
import torch

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from ignite.engine import Events, create_supervised_evaluator
from ignite.contrib.handlers.tqdm_logger import ProgressBar

# Load experiment ingredients and their respective configs.
from init.dataset import dataset, get_dataset, get_data_spliters
from init.models import model, init_model
from init.training import training, init_loader, get_n_epochs, mse_recons, \
                          slate_loss, init_loss, init_metrics, init_optimizer,\
                          init_lr_scheduler, reduce_on_plateau, token_xent, \
                          ModelCheckpoint, create_supervised_trainer

from init.configs import models as model_params
from init.configs import training as train_params


# Set up experiment
ex = Experiment(name='discovery', ingredients=[dataset, model, training])

# Required for ProgressBar to work properly
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Observers
# ex.observers.append(FileStorageObserver.create('data/sims/discovery'))
ex.observers.append(FileStorageObserver.create('data/sims/generalisation'))

# General configs
ex.add_config(no_cuda=False, temp_folder='data/temp/discovery')
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


def loss_string(name, output):
    return {f"{name.upper()}": f"{output:,.2f}"}


def metrics_string(metrics):
    output = ['Validation:']
    for k, v in metrics.items():
        output.append(f"{k.upper()}={v:,.2f}")
    return ' '.join(output)


def aggregate_metrics(metrics):
    return sum(v for v in metrics.values())


# Dataset configs
dataset.add_config(setting='unsupervised', shuffle=True)


# Model configs
# model.config(model_params.slate)
model.named_config(model_params.slate)
# model.named_config(model_params.patch_dvae)
model.named_config(model_params.slot_ae)
model.named_config(model_params.medium_ae)
model.named_config(model_params.small_ae)
model.named_config(model_params.large_ae)


# Training configs
slate_metrics = [mse_recons, token_xent]

training.add_config(loss=slate_loss, metrics=slate_metrics, scheduler=None)
training.add_named_config('slate_loss', loss=slate_loss, metrics=slate_metrics)
training.add_named_config('mse_recons', loss=mse_recons, metrics=[mse_recons])

# training.config(train_params.slate)
training.named_config(train_params.slate)
training.named_config(train_params.slot_ae)
training.named_config(train_params.reduce_lr_on_plateau)
training.named_config(train_params.exponential_lr)
training.named_config(train_params.warmup)


# Run experiment
@ex.automain
def main(temp_folder):
    device = set_seed_and_device()

    data_filters = get_data_spliters()
    dataset = get_dataset(data_filters=data_filters, train=True)

    training_loader, validation_loader = init_loader(dataset)

    # Init model
    img_size = training_loader.dataset.img_size
    model = init_model()(input_size=img_size).to(device=device)

    # Init metrics
    loss, metrics = init_loss(), init_metrics()

    # Init optimizer
    optimizer = init_optimizer(model=model)

    # Learning rate scheduling
    scheduler = init_lr_scheduler(optimizer)
    reduce_lr_on_plateau = reduce_on_plateau(optimizer)

    # Init engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    ProgressBar().attach(trainer, output_transform=partial(loss_string, 'loss'))

    validator = create_supervised_evaluator(model, metrics, device=device)
    n_val_iterations = np.ceil(len(validation_loader) * 0.2)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_loader, epoch_length=n_val_iterations)
        validator.logger.info(metrics_string(validator.state.metrics))

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.terminate()

    # Update lr
    if scheduler is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  lambda _: scheduler.step())

    if reduce_lr_on_plateau is not None:
        @validator.on(Events.EPOCH_COMPLETED)
        def reduce_lr(engine):
            reduce_lr_on_plateau.step(aggregate_metrics(engine.state.metrics))

    # Record training progression
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        loss = engine.state.output
        ex.log_scalar('training_loss', loss)

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    best_checkpoint = ModelCheckpoint(
        dirname=temp_folder,
        filename_prefix='slate_best',
        score_function=lambda e: -aggregate_metrics(e.state.metrics),
        create_dir=True,
        require_empty=False,
    )
    validator.add_event_handler(Events.COMPLETED, best_checkpoint,
                                {'model': model})

    # Run the training
    trainer.run(training_loader, max_epochs=get_n_epochs(training_loader))

    # Select best model
    model.load_state_dict(best_checkpoint.last_checkpoint_state)

    # # Save best model performance and state
    ex.add_artifact(best_checkpoint.last_checkpoint, 'trained-model.pt')
