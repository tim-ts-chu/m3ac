
import math
import torch
import torch.distributions as td
import logging
from typing import List
from agent.models.policy import MLP
from replay.replay import BufferFields

class DiscriminateAgent:
    def __init__(self,
            device_id: int,
            hidden_size: List):

        self._device_id = device_id

        self._disc_mlp = MLP(
                BufferFields['state']+BufferFields['action']+BufferFields['reward']+\
                        BufferFields['done']+BufferFields['next_state'],
                hidden_size, 1).to(device_id)
    
    def params(self):
        return self._disc_mlp.parameters()

    def discriminate(self, state, action, reward, done, next_state):
        done = done.to(self._device_id)
        next_state = next_state.to(self._device_id)
        x = torch.cat((state, action, reward, done, next_state), dim=1).to(self._device_id)
        logits = self._disc_mlp(x)
        pred = logits.clone()
        pred[pred>0] = 1
        pred[pred<=0] = 0
        return logits, pred 

