
import math
import torch
import torch.distributions as td
import logging
from typing import List
from agent.models.mlp import WorldMlp
from replay.replay import BufferFields

class DiscriminateAgent:
    def __init__(self,
            device_id: int,
            hidden_size: List,
            activation: torch.nn.Module):

        self._device_id = device_id

        self._disc_mlp = WorldMlp(
                BufferFields['state'] + BufferFields['action'] + \
                        BufferFields['done'] + BufferFields['reward'] + BufferFields['next_state'],
                hidden_size, 1, activation).to(device_id)
    
    def params(self):
        return self._disc_mlp.parameters()

    def discriminate(self, state, action, reward, done, next_state):
        x = torch.cat((state, action, reward, done, next_state), dim=1)
        logits = self._disc_mlp(x)
        pred = logits.clone()
        pred[pred>0] = 1
        pred[pred<=0] = 0
        return logits, pred 

