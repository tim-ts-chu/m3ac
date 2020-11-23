
import torch
import torch.nn.functional as F
from typing import Dict
from envs.fake_env import BaseFakeEnv
from replay.replay import ReplayBuffer, BufferFields
from agent.disc_agent import DiscriminateAgent
from agent.model_agent import ModelAgent
from agent.sac_agent import SACAgent

class DiscriminateAlgorithm:
    def __init__(self,
            device_id: int,
            real_buffer: ReplayBuffer,
            imag_buffer: ReplayBuffer,
            disc_agent: DiscriminateAgent,
            model_agent: ModelAgent,
            policy_agent: SACAgent,
            fake_env):

        self._device_id = device_id
        self._real_buffer = real_buffer
        self._imag_buffer = imag_buffer
        self._disc_agent = disc_agent
        self._model_agent = model_agent
        self._policy_agent = policy_agent
        self._fake_env = fake_env

        self._disc_optimizer = torch.optim.Adam(disc_agent.params(), lr=1e-4)

        # TODO should we use the optimizer which is different from the leaning model one
        self._transition_optimizer = torch.optim.Adam(model_agent.transition_params(), lr=1e-4)
        self._reward_optimizer = torch.optim.Adam(model_agent.reward_params(), lr=1e-4)
        #self._done_optimizer = torch.optim.Adam(model_agent.done_params(), lr=1e-4)

    def optimize_agent(self, batch_size, num_iter: int, step: int) -> Dict:
        optim_info = {}
        optim_info['discError'] = []
        optim_info['discLoss'] = []

        for it in range(num_iter):
            real_samples = self._real_buffer.sample(batch_size, self._device_id)
            #imag_samples = self._imag_buffer.sample(batch_size, self._device_id)

            with torch.no_grad():
                # generate imaginary sample
                imag_samples = {}
                # next_state_diff_dist = self._model_agent.transition(real_samples['state'], real_samples['action'])
                # next_state_sample = real_samples['state']+next_state_diff_dist.sample()
                next_state, reward, done, info = self._fake_env.step(real_samples['state'], real_samples['action'])
                # reward_dist = self._model_agent.reward(real_samples['state'], real_samples['action'])
                # reward_sample = reward_dist.sample()
                # done_logits, done_pred = self._model_agent.done(real_samples['state'], real_samples['action'])
                # done_sample = done_pred
                imag_samples['state'] = real_samples['state']
                imag_samples['action'] = real_samples['action']
                # imag_samples['reward'] = reward_sample
                # imag_samples['done'] = done_sample
                imag_samples['reward'] = real_samples['reward']
                imag_samples['done'] = real_samples['done']
                imag_samples['next_state'] = next_state

            samples = {}
            for k in BufferFields.keys():
                samples[k] = torch.cat((real_samples[k], imag_samples[k]), dim=0)
            labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size))).view(-1,1).to(self._device_id)

            self._disc_optimizer.zero_grad()
            logits, pred = self._disc_agent.discriminate(
                    samples['state'].detach(),
                    samples['action'].detach(),
                    samples['reward'].detach(),
                    samples['done'].detach(),
                    samples['next_state'].detach())
            error = (pred-labels).abs().sum()/(2*batch_size)

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            self._disc_optimizer.step()

            optim_info['discError'].append(error)
            optim_info['discLoss'].append(loss)

        return optim_info

    def optimize_model_agent(self, batch_size, step: int):
        optim_info = {}
        real_samples = self._real_buffer.sample(batch_size, self._device_id)

        self._transition_optimizer.zero_grad()
        self._reward_optimizer.zero_grad()
        self._done_optimizer.zero_grad()

        # generate imaginary sample
        next_state_diff_dist = self._model_agent.transition(real_samples['state'], real_samples['action'])
        reward_dist = self._model_agent.reward(real_samples['state'], real_samples['action'])
        done_logits, done_pred = self._model_agent.done(real_samples['state'], real_samples['action'])
        next_state_sample = real_samples['state']+next_state_diff_dist.rsample()
        # reward_sample = reward_dist.rsample()
        # done_sample = done_pred
        reward_sample = real_samples['reward']
        done_sample = real_samples['done']

        random_action = torch.rand(batch_size, BufferFields['action'], device=self._device_id)*2-1
        logits, pred = self._disc_agent.discriminate(
                real_samples['state'],
                real_samples['action'], # TODO how should we use the record action? or use current policy to decide?
                #random_action, # TODO how should we use the record action? or use current policy to decide?
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


