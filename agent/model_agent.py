
import math
import torch
import torch.distributions as td
import logging
from typing import List
from agent.models.policy import MLP
from replay.replay import BufferFields
from torch.distributions.multivariate_normal import MultivariateNormal

class ModelAgent:

    def __init__(self,
            device_id: int,
            model_hidden_size: List,
            reward_hidden_size: List,
            done_hidden_size: List):

        self._device_id = device_id

        # output gaussian mean, std
        self._transition_mlp = MLP(
                BufferFields['state']+BufferFields['action'],
                model_hidden_size,
                BufferFields['state']*2).to(device_id)

        # output gaussian mean, std
        self._reward_mlp = MLP(
                BufferFields['state']+BufferFields['action'],
                reward_hidden_size,
                2).to(device_id)

        # output sigmoid 
        self._done_mlp = MLP(
                BufferFields['state']+BufferFields['action'],
                done_hidden_size,
                1).to(device_id)

    def transition_params(self):
        return self._transition_mlp.parameters()

    def reward_params(self):
        return self._reward_mlp.parameters()

    def done_params(self):
        return self._done_mlp.parameters()

    def transition(self, state, action):
        out = self._transition_mlp(torch.cat((state, action), dim=1))
        loc = out[:, :BufferFields['state']]
        std = torch.exp(out[:, BufferFields['state']:])
        normal = td.normal.Normal(loc, std)
        dist = td.independent.Independent(normal, 1)
        return dist 
        next_state = dist.rsample()
        return next_state

    def reward(self, state, action):
        out = self._reward_mlp(torch.cat((state, action), dim=1))
        loc = out[:, :BufferFields['reward']]
        std = torch.exp(out[:, BufferFields['reward']:])
        normal = td.normal.Normal(loc, std)
        dist = td.independent.Independent(normal, 1)
        return dist
        reward = dist.rsample()
        return reward 

    def done(self, state, action, get_logit=True):
        logits = self._done_mlp(torch.cat((state, action), dim=1))
        pred = logits.clone()
        pred[pred>0] = True
        pred[pred<=0] = False
        return logits, pred 




