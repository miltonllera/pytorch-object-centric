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
        return self.latent(self.encoder(inputs))

    def forward(self, inputs):
        z = self.embed(inputs)
        recons = self.decoder(z)
        return recons, z


class VariationalAutoEncoder(AutoEncoder):
    def embed(self, inputs):
        return self.latent(self.encoder(inputs))[0]  # only return embedding

    def forward(self, inputs):
        z = self.latent(self.encoder(inputs))
        recons = self.decoder(z[0])  # only use the embedding z
        return recons, z
