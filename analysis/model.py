import sys
import os
import yaml
from os import path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sacred import Experiment

from init.dataset import dataset, get_lazyloader
from init.training import training, init_metrics, init_loss, \
                          mse_recons, bern_recons
from init.analysis import analysis, infer, model_score, generate_recons, \
                          plot_recons, compute_slot_masks, plot_masks, \
                          learning_curve_plot

from init.models import load_slot_ae


an = Experiment(name='analysis', ingredients=[analysis, dataset, training])


an.add_config(scores=True, recons=True, learning_curve=True,
              root_folder='data/results', no_cuda=False)

# Configs to run only one analysis
an.add_named_config('learning', plot_recons=False, score_model=False)
an.add_named_config('score', plot_recons=False, plot_learning_curve=False)
an.add_named_config('recons', score_model=False, plot_learning_curve=False)

# Run all but X
an.add_named_config('noscore', score_model=False)
an.add_named_config('norecons', plot_recons=False)

# Run either performance or latent representation plots
an.add_named_config('perfm', compute_disent=False, plot_latent_rep=False)

training.add_named_config('recons_mse', metrics=[mse_recons])
training.add_named_config('recons_bern', metrics=[bern_recons])


def is_generative(setting):
    return setting in ['composition', 'unsupervised', 'recons']


@an.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def load_model(setting, dataset, model_folder, device):
    if setting == 'unsupervised':
        model = load_slot_ae(dataset.img_size, model_folder)
    else:
        raise ValueError()
    return model.to(device=device)


@an.automain
def main(model_id, exp_folder, scores, learning_curve, recons, root_folder):

    print('Running analysis for model {}.'.format(model_id))

    device = set_seed_and_device()

    model_folder = path.join(exp_folder, str(model_id))

    with open(path.join(model_folder, 'config.json')) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Extract dataset conifg
    dataset_config = configs['dataset']

    dataset = dataset_config['dataset']
    setting = dataset_config['setting']
    condition = dataset_config.get('condition', None)
    variant = dataset_config.get('variant', None)
    modifiers = dataset_config.get('modifiers', None)

    if is_generative(setting):
        loss = configs['training']['loss']['params']['loss']
    else:
        loss = configs['training']['loss']['name']

    # Create results folder
    results_folder = path.join(root_folder, setting, str(model_id))
    os.makedirs(results_folder, exist_ok=True)

    if learning_curve:
        print('Plotting learning curve')
        fig = learning_curve_plot(model_folder)
        fig.savefig(path.join(results_folder, 'learning_curve'),
                    bbox_inches='tight')
        print('Done')


    if not (scores or recons):
        print('Analysis finished')
        return

    print('Loading model and dataset...')

    # model = load_composer(model_folder, model_id, device)
    dataset = get_lazyloader(dataset, condition, variant, modifiers)
    model = load_model(setting, dataset, model_folder, device)

    print('Done.')

    if scores:
        print('Computing model scores...')
        if loss == 'mse':
            if is_generative(setting):
                metrics = init_metrics([mse_recons])
            else:
                metrics = create_r_squares(dataset.factors)
        elif loss == 'bce':
            metrics = init_metrics([bern_recons])
        else:
            raise ValueError("Unsuported loss {}.".format(loss))

        print('Scoring training data...')
        train_score = model_score(model,
                dataset.get_unsupervised(train=True), metrics=metrics)

        print('Done.')

        test_data = dataset.get_unsupervised(train=False)

        if test_data is not None:
            print('Test data provided. Scoring model on OOD samples...')

            test_score = model_score(model,
                    dataset.get_unsupervised(train=False), metrics=metrics)

            all_scores = pd.concat([train_score, test_score],
                                   keys=['Train', 'Test'], names=['Data'])

            print('Done.')
        else:
            print('No test data provided')
            all_scores = train_score.reset_index()

        all_scores.to_csv(path.join(results_folder, 'scores.csv'))

        print('Saving results.\nDone.')

    train_data = dataset.get_supervised(train=True, pred_type='reg')
    test_data = dataset.get_supervised(train=False, pred_type='reg')

    if is_generative(setting) and recons:
        print('Generating reconstruction examples for training data...')

        train_recons  = generate_recons(model, train_data, loss=loss)[:2]
        masks = compute_slot_masks(model, train_data)[:2]

        train_recons_fig = plot_recons(train_recons)
        mask_fig = plot_masks(masks)

        train_recons_fig.savefig(path.join(results_folder, 'train_recons.png'),
                                 bbox_inches='tight')
        mask_fig.savefig(path.join(results_folder, 'mask.png'),
                         bbox_inches='tight')

        print('Done.')

        if test_data is not None:
            print('Generating reconstruction examples for test data...')

            outputs = generate_recons(model, test_data, loss=loss)
            test_recons, test_masks = outputs[:2], outputs[::2]

            test_recons_fig = recons(test_recons)
            test_mask_fig = recons(test_masks)

            test_recons_fig.savefig(path.join(results_folder,
                                              'test_recons.png'),
                                     bbox_inches='tight')
            test_mask_fig.savefig(path.join(results_folder, 'test_mask.png'),
                                  bbox_inches='tight')

            print('Done.')

    print('Analysis finished')
