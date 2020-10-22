
import torch
from typing import Dict
from replay.replay import ReplayBuffer, BufferFields
from agent.disc_agent import DiscriminateAgent
from agent.model_agent import ModelAgent

class DiscriminateAlgorithm:
    def __init__(self,
            device_id: int,
            real_buffer: ReplayBuffer,
            imag_buffer: ReplayBuffer,
            disc_agent: DiscriminateAgent,
            model_agent: ModelAgent):

        self._device_id = device_id
        self._real_buffer = real_buffer
        self._imag_buffer = imag_buffer
        self._disc_agent = disc_agent
        self._model_agent = model_agent

        self._disc_optimizer = torch.optim.Adam(disc_agent.params(), lr=1e-4)

        # TODO should we use the optimizer which is different from the leaning model one
        self._transition_optimizer = torch.optim.Adam(model_agent.transition_params(), lr=1e-4)
        self._reward_optimizer = torch.optim.Adam(model_agent.reward_params(), lr=1e-4)
        self._done_optimizer = torch.optim.Adam(model_agent.done_params(), lr=1e-4)

    def optimize_agent(self, batch_size, step: int) -> Dict:
        optim_info = {}
        real_samples = self._real_buffer.sample(batch_size, self._device_id)
        imag_samples = self._imag_buffer.sample(batch_size, self._device_id)
        samples = {}
        for k in BufferFields.keys():
            samples[k] = torch.cat((real_samples[k], imag_samples[k]), dim=0)
        labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size))).view(-1,1).to(self._device_id)

        self._disc_optimizer.zero_grad()
        logits, pred = self._disc_agent.discriminate(
                samples['state'],
                samples['action'],
                samples['reward'],
                samples['done'],
                samples['next_state'])
        error = (pred-labels).norm()/batch_size

        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        self._disc_optimizer.step()

        optim_info['discError'] = error
        optim_info['discLoss'] = loss

        return optim_info

    def optimize_model_agent(self, batch_size, step: int):
        optim_info = {}
        real_samples = self._real_buffer.sample(batch_size, self._device_id)

        # imagine samples
        self._transition_optimizer.zero_grad()
        self._reward_optimizer.zero_grad()
        self._done_optimizer.zero_grad()

        next_state_dist = self._model_agent.transition(real_samples['state'], real_samples['action'])
        reward_dist = self._model_agent.reward(real_samples['state'], real_samples['action'])
        done_logits, done_pred = self._model_agent.done(real_samples['state'], real_samples['action'])
        next_state_sample = next_state_dist.rsample()
        reward_sample = reward_dist.rsample()
        done_sample = done_pred

        logits, pred = self._disc_agent.discriminate(
                real_samples['state'],
                real_samples['action'], # TODO how should we use the record action? or use current policy to decide?
                reward_sample,
                done_sample, # FIXME how to backprop?
                next_state_sample)

        labels = torch.ones(batch_size).view(-1,1).to(self._device_id)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()

        self._transition_optimizer.step()
        self._reward_optimizer.step()
        self._done_optimizer.step()

        optim_info['discModelLoss'] = loss

        return optim_info


