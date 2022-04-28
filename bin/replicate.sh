# !/bin/bash

# Slot Attention Autoencder (Locatello et al.)
python -m experiments.discovery with dataset.tetrominoes \
                                  model.{slot_ae,small_ae}  \
                                  training.{mse_recons,slot_ae} \
                                  training.{warmup,warmup_groups=None}

python -m experiments.discovery with dataset.shapes3d \
                                  model.slot_ae  \
                                  training.{mse_recons,slot_ae} \
                                  training.{warmup,warmup_groups=None}


# use implicit gradient in slot attention (Chang et al)
python -m experiments.discovery with dataset.shapes3d \
                                  model.{slot_ae,n_slots=3}  \
                                  model.approx_implicit_grad=True \
                                  training.{mse_recons,slot_ae} \
                                  training.{warmup,warmup_groups=None}


# # SLATE (Singh et al)
python -m experiments.discovery with dataset.shapes3d \
                                  model.approx_implicit_grad=False \
                                  training.{reduce_lr_on_plateau,warmup} \
                                  training.grad_norm=1.0


# use implicit gradient in slot attention (Chang et al)
python -m experiments.discovery with dataset.shapes3d \
                                  model.{slate,n_slots=3} \
                                  model.approx_implicit_grad=True \
                                  training.{slate,reduce_lr_on_plateau,warmup}
