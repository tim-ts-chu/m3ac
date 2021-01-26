
import math
import torch
import torch.distributions as td
import logging
from typing import List
from agent.models.policy import MLP
from agent.models.mlp import WorldMlp
from replay.replay import BufferFields

class ModelAgent:

    def __init__(self,
            device_id: int,
            trans_hidden_size: List,
            reward_hidden_size: List,
            model_activation: torch.nn.Module,
            use_batchnorm: bool,
            dropout_prob: float=None,
            deterministic: bool=False):

        self._deterministic = deterministic
        self._device_id = device_id
        self._logger = logging.getLogger()

        # output gaussian mean, std
        self._transition_mlp = WorldMlp(
                BufferFields['state']+BufferFields['action'],
                trans_hidden_size,
                BufferFields['state']*2,
                activation=model_activation,
                use_batchnorm=use_batchnorm,
                dropout_prob=dropout_prob,
                ).to(device_id)

        # output gaussian mean, std
        self._reward_mlp = WorldMlp(
                BufferFields['state']+BufferFields['action']+BufferFields['next_state'],
                reward_hidden_size,
                2,
                activation=model_activation,
                use_batchnorm=use_batchnorm,
                dropout_prob=dropout_prob,
                ).to(device_id)

        self._logger.info('Transition Model:')
        self._logger.info(self._transition_mlp)
        self._logger.info('Reward Model:')
        self._logger.info(self._reward_mlp)

    def transition_params(self):
        return self._transition_mlp.parameters()

    def reward_params(self):
        return self._reward_mlp.parameters()

    def done_params(self):
        return self._done_mlp.parameters()

    def eval(self):
        self._transition_mlp.eval()
        self._reward_mlp.eval()

    def train(self):
        self._transition_mlp.train()
        self._reward_mlp.train()
        self._done_mlp.train()

    def transition(self, state, action):
        out = self._transition_mlp(torch.cat((state, action), dim=1))
        loc = out[:, :BufferFields['state']]
        if self._deterministic:
            return loc
        else:
            std = torch.exp(out[:, BufferFields['state']:])
            normal = td.normal.Normal(loc, std)
            dist = td.independent.Independent(normal, 1)
            return dist.rsample()

    def reward(self, state, action, next_state):
        out = self._reward_mlp(torch.cat((state, action, next_state), dim=1))
        loc = out[:, :BufferFields['reward']]
        if self._deterministic:
            return loc
        else:
            std = torch.exp(out[:, BufferFields['reward']:])
            normal = td.normal.Normal(loc, std)
            dist = td.independent.Independent(normal, 1)
            return dist.rsample()

