
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
                # BufferFields['state']*2+BufferFields['action']*2,
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

    def set_data_statistics(self, state, action, next_state):
        # FIXME not use currently
        self._state_mean = state.mean(dim=0)
        self._state_std = state.std(dim=0)

        self._action_mean = action.mean(dim=0)
        self._action_std = action.std(dim=0)

        self._next_state_mean = next_state.mean(dim=0)
        self._next_state_std = next_state.std(dim=0)

    def preprocess(self, state, action, next_state):
        # FIXME not use currently
        normalized_state = (state - self._state_mean) / self._state_std
        normalized_action = (action - self._action_mean) / self._action_std
        normalized_next_state = (next_state - self._next_state_mean) / self._next_state_std
        return normalized_state, normalized_action, normalized_next_state

    def postprocess(self, next_state):
        # FIXME not use currently
        post_next_state = next_state * self._next_state_std + self._next_state_mean
        return post_next_state

    # def transition(self, state, state_add, action, action_add, ret_logvar=False):
        # out = self._transition_mlp(torch.cat((state, state_add, action, action_add), dim=1))
    def transition(self, state, action, ret_logvar=False):
        out = self._transition_mlp(torch.cat((state, action), dim=1))
        loc = out[:, :BufferFields['state']]

        if ret_logvar:
            return out[:, :BufferFields['state']], out[:, BufferFields['state']:]

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

