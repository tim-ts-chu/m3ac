

import torch
import torch.nn.functional as F
from typing import Dict
from envs.fake_env import BaseFakeEnv
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
            disc_agent: DiscriminateAgent,
            fake_env: BaseFakeEnv,
            transition_reg_loss_weight: float,
            transition_gan_loss_weight: float,
            reward_reg_loss_weight: float,
            reward_gan_loss_weight: float):

        self._device_id = device_id
        self._real_buffer = real_buffer
        self._imag_buffer = imag_buffer
        self._model_agent = model_agent
        self._policy_agent = policy_agent
        self._disc_agent = disc_agent
        self._fake_env = fake_env

        self._transition_reg_loss_weight = transition_reg_loss_weight
        self._transition_gan_loss_weight = transition_gan_loss_weight
        self._reward_reg_loss_weight = reward_reg_loss_weight
        self._reward_gan_loss_weight = reward_gan_loss_weight

        self._transition_optimizer = torch.optim.Adam(model_agent.transition_params(), lr=1e-4)
        self._reward_optimizer = torch.optim.Adam(model_agent.reward_params(), lr=1e-4)
        self._done_optimizer = torch.optim.Adam(model_agent.done_params(), lr=1e-4)

    def optimize_agent(self, batch_size: int, num_iter: int, step: int) -> Dict:
        optim_info = {}
        optim_info['transitionRegLoss'] = []
        optim_info['transitionGanLoss'] = []
        optim_info['rewardRegLoss'] = []
        optim_info['rewardGanLoss'] = []
        optim_info['doneLoss'] = []

        optim_info['transitionError'] = []
        optim_info['rewardError'] = []
        optim_info['doneError'] = []

        # calculate multi-step error
        # max_steps = 20
        # multisteps_errors = {}
        # sample_seq = self._real_buffer.sample_sequence(batch_size, max_steps, self._device_id)
        # current_state = sample_seq['state'][:, 0, :]
        # done_mask = torch.ones(batch_size, 1, device=self._device_id) # (b, 1)
        # with torch.no_grad():
            # for t in range(max_steps):
                # mu, log_std, action, log_pi = self._policy_agent.pi(current_state)
                # next_state_diff_dist = self._model_agent.transition(current_state, action)
                # next_state_pred = current_state + next_state_diff_dist.sample()

                # done_mask[sample_seq['done'][:, t, :]>0.5] = 0  # (b, 1)
                # next_state_real = sample_seq['next_state'][:, t, :]
                # square_errors = (next_state_real-next_state_pred).square()*done_mask
                # if torch.sum(done_mask) > 0.5:
                    # multisteps_errors[t] = torch.clamp(square_errors, 0, 1000).sum()/torch.sum(done_mask)
                # else:
                    # multisteps_errors[t] = 0
                
                # current_state = next_state_pred

        # optim_info['transitionError-1'] = multisteps_errors[0]
        # optim_info['transitionError-3'] = multisteps_errors[2]
        # optim_info['transitionError-5'] = multisteps_errors[4]
        # optim_info['transitionError-10'] = multisteps_errors[9]
        # optim_info['transitionError-20'] = multisteps_errors[19]
        
        for it in range(num_iter):
            real_samples = self._real_buffer.sample(batch_size, self._device_id)

            self._transition_optimizer.zero_grad()
            self._reward_optimizer.zero_grad()
            self._done_optimizer.zero_grad()
            
            # fake env will clamp abnormal state
            #next_state, reward, done, _ = self._fake_env.step(real_samples['state'], real_samples['action'])

            # transition loss
            next_state_diff_dist = self._model_agent.transition(real_samples['state'], real_samples['action'])
            next_state_pred = real_samples['state'] + next_state_diff_dist.rsample()
            next_state_error = (next_state_pred - real_samples['next_state']).abs().sum()/batch_size
            optim_info['transitionError'].append(next_state_error)

            transition_reg_loss = F.mse_loss(next_state_pred, real_samples['next_state'])

            # logits, pred = self._disc_agent.discriminate(
                    # real_samples['state'],
                    # real_samples['action'], # TODO how should we use the record action? or use current policy to decide?
                    # real_samples['reward'],
                    # real_samples['done'], # FIXME how to backprop?
                    # next_state_sample)
            # labels = torch.ones(batch_size).view(-1,1).to(self._device_id)
            # transition_gan_loss = F.binary_cross_entropy_with_logits(logits, labels)
            transition_gan_loss = 0

            optim_info['transitionRegLoss'].append(transition_reg_loss)
            optim_info['transitionGanLoss'].append(transition_gan_loss)

            # reward loss
            if not self._model_agent.reward:
                reward_reg_loss = 0
                reward_gan_loss = 0
            else:
                reward_dist = self._model_agent.reward(
                        real_samples['state'], real_samples['action'], real_samples['next_state'])
                reward_pred = reward_dist.rsample()
                reward_error = (reward_pred - real_samples['reward']).abs().sum()/batch_size
                optim_info['rewardError'].append(reward_error)
                reward_reg_loss = F.mse_loss(reward_pred, real_samples['reward'])
                reward_gan_loss = 0 # TODO only discriminate on transition at this point

            optim_info['rewardRegLoss'].append(reward_reg_loss)
            optim_info['rewardGanLoss'].append(reward_gan_loss)

            # done loss
            if not self._model_agent.done:
                done_loss = torch.tensor(0.0, requires_grad=True)
            else:
                done_logits, done_pred = self._model_agent.done(
                        real_samples['state'], real_samples['action'], real_samples['next_state'])
                done_error = (done_pred - real_samples['done']).abs().sum()/batch_size
                optim_info['doneError'].append(done_error)
                done_loss = F.binary_cross_entropy_with_logits(done_logits, real_samples['done'])
            
            optim_info['doneLoss'].append(done_loss)

            # weighted sum total loss
            # total_loss = \
                    # self._transition_reg_loss_weight * transition_reg_loss + \
                    # self._reward_reg_loss_weight * reward_reg_loss + \
                    # self._transition_gan_loss_weight * transition_gan_loss + \
                    # self._reward_gan_loss_weight * reward_gan_loss + \
                    # done_loss

            # total_loss.backward()
            transition_reg_loss.backward()
            reward_reg_loss.backward()
            done_loss.backward()

            self._transition_optimizer.step()
            self._reward_optimizer.step()
            self._done_optimizer.step()

        return optim_info

    def generate_samples(self, batch_size: int):
        real_samples = self._real_buffer.sample(batch_size, self._device_id)
        state = real_samples['state']
        with torch.no_grad():
            _, _, action, _ = self._policy_agent.pi(state)
            next_state, reward, done, info = self._fake_env.step(state, action)

            samples = {}
            samples['state'] = state
            samples['action'] = action
            samples['reward'] = reward
            samples['done'] = done
            samples['next_state'] = next_state

        return samples

