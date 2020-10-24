

import torch
from typing import Dict
from replay.replay import ReplayBuffer
from agent.model_agent import ModelAgent
from agent.sac_agent import SACAgent

class ModelAlgorithm:

    def __init__(self,
            device_id: int,
            real_buffer: ReplayBuffer,
            imag_buffer: ReplayBuffer,
            model_agent: ModelAgent,
            policy_agent: SACAgent):

        self._device_id = device_id
        self._real_buffer = real_buffer
        self._imag_buffer = imag_buffer
        self._model_agent = model_agent
        self._policy_agent = policy_agent

        self._transition_optimizer = torch.optim.Adam(model_agent.transition_params(), lr=1e-4)
        self._reward_optimizer = torch.optim.Adam(model_agent.reward_params(), lr=1e-4)
        self._done_optimizer = torch.optim.Adam(model_agent.done_params(), lr=1e-4)

    def optimize_agent(self, batch_size: int, num_iter: int, step: int) -> Dict:
        optim_info = {}
        optim_info['transitionLoss'] = []
        optim_info['rewardLoss'] = []
        optim_info['doneLoss'] = []
        optim_info['transitionError'] = []
        optim_info['rewardError'] = []
        optim_info['doneError'] = []
        
        for it in range(num_iter):
            real_samples = self._real_buffer.sample(batch_size, self._device_id)

            self._transition_optimizer.zero_grad()
            self._reward_optimizer.zero_grad()
            self._done_optimizer.zero_grad()
            
            next_state_dist = self._model_agent.transition(real_samples['state'], real_samples['action'])
            reward_dist = self._model_agent.reward(real_samples['state'], real_samples['action'])
            done_logits, done_pred = self._model_agent.done(real_samples['state'], real_samples['action'])

            # imagine next_state, reward, done?
            next_state_sample = next_state_dist.sample()
            reward_sample = reward_dist.sample()
            done_sample = done_pred

            # self._imag_buffer.push_batch(
                # state=real_samples['state'].detach(),
                # action=real_samples['action'].detach(),
                # reward=reward_sample.detach(),
                # done=done_sample.detach(),
                # next_state=next_state_sample.detach())

            next_state_error = (next_state_sample-real_samples['next_state']).norm()/batch_size
            reward_error = (reward_sample-real_samples['reward']).norm()/batch_size
            done_error = (done_pred-real_samples['done']).norm()/batch_size

            optim_info['transitionError'].append(next_state_error)
            optim_info['rewardError'].append(reward_error)
            optim_info['doneError'].append(done_error)

            transition_loss = -torch.mean(next_state_dist.log_prob(real_samples['next_state']))
            reward_loss = -torch.mean(reward_dist.log_prob(real_samples['reward']))
            done_loss = torch.nn.BCEWithLogitsLoss()(done_logits, real_samples['done'])

            transition_loss.backward()
            reward_loss.backward()
            done_loss.backward()

            self._transition_optimizer.step()
            self._reward_optimizer.step()
            self._done_optimizer.step()
            
            optim_info['transitionLoss'].append(transition_loss)
            optim_info['rewardLoss'].append(reward_loss)
            optim_info['doneLoss'].append(done_loss)

        return optim_info

