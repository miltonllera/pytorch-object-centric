# Slot Attention in PyTorch

PyTorch implementation of slot-attention-based architectures for object discover. This code uses [Sacred](https://sacred.readthedocs.io/en/stable/) to define and run experiments.

## Models implemented

Two models are provided, the standard Slot-Attention model ([Locatello et al, 2020](https://github.com/google-research/google-research/tree/master/slot_attention)) and SLATE ([Singh et al, 2022](https://github.com/singhgautam/slate)). Additionally, users can use implicit gradients as proposed in ([Chang et al, 2022](https://openreview.net/pdf?id=SSgxDBOIqgq)).

## Running experiments

To replicate the experiments in the above articles, users should run the script ``bin/replicate`` inside the ``experiments`` folder. Only ``3DShapes`` and ``Tetrominoes`` datasets are provided, with raw data found at their respective repos.

The environment can be installed using the following command:

```
conda env create -f torchlab-env.yml
```

The relevant ``.yaml`` config file can be found [here](https://gist.github.com/miltonllera/e0a6ca7f3283b029d0e333730b0ce980#file-torchlab-yaml)

## Attributions

We thank the authors of both the Slot-Attention and Slate models for making their code available, wich allowed us to reproduce the results.

## Citation

```bibtex
@misc{locatello2020objectcentric,
    title = {Object-Centric Learning with Slot Attention},
    author = {Francesco Locatello and Dirk Weissenborn and Thomas Unterthiner and Aravindh Mahendran and Georg Heigold and Jakob Uszkoreit and Alexey Dosovitskiy and Thomas Kipf},
    year = {2020},
    eprint = {2006.15055},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}

@inproceedings{
      singh2022illiterate,
      title={Illiterate DALL-E Learns to Compose},
      author={Gautam Singh and Fei Deng and Sungjin Ahn},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=h0OYV0We3oh}
}
```
