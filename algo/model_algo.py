

import torch
from typing import Dict
from replay.replay import ReplayBuffer
from agent.model_agent import ModelAgent
from agent.sac_agent import SACAgent
from agent.disc_agent import DiscriminateAgent

class ModelAlgorithm:

    def __init__(self,
            device_id: int,
            real_buffer: ReplayBuffer,
            imag_buffer: ReplayBuffer,
            model_agent: ModelAgent,
            policy_agent: SACAgent,
            disc_agent: DiscriminateAgent):

        self._device_id = device_id
        self._real_buffer = real_buffer
        self._imag_buffer = imag_buffer
        self._model_agent = model_agent
        self._policy_agent = policy_agent
        self._disc_agent = disc_agent

        self._transition_loss_weight = 0
        self._reward_loss_weight = 0
        self._done_loss_weight = 0
        self._disc_loss_weight = 1

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
            
            next_state_diff_dist = self._model_agent.transition(real_samples['state'], real_samples['action'])
            reward_dist = self._model_agent.reward(real_samples['state'], real_samples['action'])
            done_logits, done_pred = self._model_agent.done(real_samples['state'], real_samples['action'])

            # imagine next_state, reward, done?
            next_state_sample = real_samples['state'] + next_state_diff_dist.rsample()
            # reward_sample = reward_dist.rsample()
            # done_sample = done_pred
            reward_sample = real_samples['reward']
            done_sample = real_samples['done']

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

            # model loss (mean square error)
            transition_loss = torch.mean((next_state_sample-real_samples['next_state']).square())
            reward_loss = torch.mean((reward_sample-real_samples['reward']).square())
            done_loss = torch.nn.BCEWithLogitsLoss()(done_logits, real_samples['done'])
            
            # discriminator loss
            logits, pred = self._disc_agent.discriminate(
                    real_samples['state'],
                    real_samples['action'], # TODO how should we use the record action? or use current policy to decide?
                    #random_action, # TODO how should we use the record action? or use current policy to decide?
                    reward_sample,
                    done_sample, # FIXME how to backprop?
                    next_state_sample)

            labels = torch.ones(batch_size).view(-1,1).to(self._device_id)
            disc_loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
            optim_info['discModelLoss'] = disc_loss


            # weighted sum total loss
            total_loss = self._transition_loss_weight*transition_loss + self._reward_loss_weight*reward_loss + \
                    self._done_loss_weight*done_loss + self._disc_loss_weight*disc_loss

            total_loss.backward()

            self._transition_optimizer.step()
            self._reward_optimizer.step()
            self._done_optimizer.step()
            
            optim_info['transitionLoss'].append(transition_loss)
            optim_info['rewardLoss'].append(reward_loss)
            optim_info['doneLoss'].append(done_loss)

        return optim_info

