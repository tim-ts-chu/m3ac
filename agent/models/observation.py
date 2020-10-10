import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class ObservationEncoder(nn.Module):
    def __init__(self, obs_size, embed_size, activation=nn.ReLU):
        super().__init__()

        self._obs_size = obs_size
        self._embed_size = embed_size

        self.encoder = nn.Sequential(
            nn.Linear(self._obs_size, 256),
            activation(),
            nn.Linear(256, 256),
            activation(),
            nn.Linear(256, self._embed_size),
        )

    def forward(self, obs):
        batch_shape = obs.shape[:-1]
        embed = self.encoder(obs.view(-1, self._obs_size))
        return embed.view(*batch_shape, -1)

    @property
    def embed_size(self):
        return self._embed_size

class ObservationDecoder(nn.Module):
    def __init__(self, obs_size, embed_size, activation=nn.ReLU):
        super().__init__()

        self._obs_size = obs_size
        self._embed_size = embed_size

        self.decoder = nn.Sequential(
            nn.Linear(self._embed_size, 256),
            activation(),
            nn.Linear(256, 256),
            activation(),
            nn.Linear(256, self._obs_size),
        )

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        x = self.decoder(x.view(-1, self._embed_size))
        mean = x.view(*batch_shape, -1)
        obs_dist = td.Independent(td.Normal(mean, 1), 1)
        return obs_dist

