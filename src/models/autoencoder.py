import torch.nn as nn
from .initialization import weights_init


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, latent):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent

        self.reset_parameter()

    def reset_parameter(self):
        self.apply(weights_init)

    @property
    def n_layers(self):
        return len(self.encoder)

    @property
    def latent_size(self):
        return self.latent.size

    def embed(self, inputs):
        """Embed a batch of data points, x, into their z representations."""
        return self.latent(self.encoder(inputs))

    def forward(self, inputs):
        """
        Takes a batch of samples, encodes them, and then decodes them again.
        """
        h = self.encoder(inputs)
        z = self.latent(h)
        recons = self.decoder(z).reshape(*inputs.shape)
        return recons, z
