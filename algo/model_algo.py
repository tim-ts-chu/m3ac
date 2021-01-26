import torch
import torch.nn.functional as F
from typing import Dict
from envs.fake_env import FakeEnv
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
            fake_env: FakeEnv,
            transition_reg_loss_weight: float,
            transition_gan_loss_weight: float,
            reward_reg_loss_weight: float,
            reward_gan_loss_weight: float,
            h_step_loss: int,
            trans_lr: float,
            reward_lr: float,
            trans_weight_decay: float,
            reward_weight_decay: float,
            num_updates: int,
            model_batch_size: int):

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

        self._h_step_loss = h_step_loss
        self._num_updates = num_updates
        self._model_batch_size = model_batch_size

        self._transition_optimizer = torch.optim.Adam(
                model_agent.transition_params(),
                lr=trans_lr,
                weight_decay=trans_weight_decay)
        self._reward_optimizer = torch.optim.Adam(
                model_agent.reward_params(),
                lr=reward_lr,
                weight_decay=reward_weight_decay)

    def optimize_agent(self, step: int) -> Dict:
        optim_info = {}
        optim_info['transitionRegLoss'] = []
        optim_info['transitionGanLoss'] = []
        optim_info['rewardRegLoss'] = []
        optim_info['rewardGanLoss'] = []
        optim_info['doneLoss'] = []
        
        for it in range(self._num_updates):
            real_samples = self._real_buffer.sample_sequence(self._model_batch_size, self._h_step_loss, self._device_id)

            self._transition_optimizer.zero_grad()
            self._reward_optimizer.zero_grad()
            
            # fake env will clamp abnormal state
            next_state, reward, done, _ = self._fake_env.step(real_samples['state'][:,0,:], real_samples['action'][:,0,:])

            transition_reg_loss = F.mse_loss(next_state, real_samples['next_state'][:,0,:])
            reward_reg_loss = F.mse_loss(reward, real_samples['reward'][:,0,:])
            optim_info['transitionRegLoss'].append(transition_reg_loss)
            optim_info['rewardRegLoss'].append(reward_reg_loss)

            # n step transition loss
            # h_step_losses = torch.zeros(self._h_step_loss, device=self._device_id)
            # valid_mask = torch.ones(self._model_batch_size, device=self._device_id, requires_grad=False).bool()
            # h_state = real_samples['state'][:,0,:]
            # for h in range(self._h_step_loss):
                # valid_mask[real_samples['end'][:,h,0]>0.5] = False
                # if valid_mask.sum() < 0.5:
                    ## no more valid sequence
                    # break
                # mask = valid_mask.clone().detach()

                ## calculate loss
                # next_state_diff_dist = self._model_agent.transition(h_state, real_samples['action'][:,h,:])
                # next_state_diff = next_state_diff_dist.rsample()
                # real_state_diff = real_samples['next_state'][:,h,:] - real_samples['state'][:,h,:]
                # h_step_losses[h] = F.mse_loss(next_state_diff[mask,:], real_state_diff[mask,:])

                # h_state = h_state + next_state_diff

            # transition_reg_loss = h_step_losses.sum()/self._h_step_loss

            # gan loss
            # next_state_diff_dist = self._model_agent.transition(
                    # real_samples['state'][:,0,:], real_samples['action'][:,0,:])
            # next_state_diff = next_state_diff_dist.rsample()
            # logits, pred = self._disc_agent.discriminate(
                    # real_samples['state'][:,0,:],
                    # real_samples['action'][:,0,:],
                    # real_samples['reward'][:,0,:],
                    # real_samples['done'][:,0,:],
                    # real_samples['state'][:,0,:] + next_state_diff)
            # true_labels = torch.full((self._model_batch_size, 1), .8, dtype=logits.dtype, device=logits.device) + \
                    # torch.randn((self._model_batch_size, 1), dtype=logits.dtype, device=logits.device)/10.0
            # transition_gan_loss = F.binary_cross_entropy_with_logits(logits, true_labels)

            # optim_info['transitionRegLoss'].append(transition_reg_loss)
            # optim_info['transitionGanLoss'].append(transition_gan_loss)

            # reward loss
            # if not self._model_agent.reward:
                # reward_reg_loss = 0
                # reward_gan_loss = 0
            # else:
                # reward_dist = self._model_agent.reward(
                        # real_samples['state'][:,0,:], real_samples['action'][:,0,:], real_samples['next_state'][:,0,:])
                # reward_pred = reward_dist.rsample()
                # reward_error = (reward_pred - real_samples['reward'][:,0,:]).abs().sum()/self._model_batch_size
                # reward_reg_loss = F.mse_loss(reward_pred, real_samples['reward'][:,0,:])
                # reward_gan_loss = 0 # TODO only discriminate on transition at this point

            # optim_info['rewardRegLoss'].append(reward_reg_loss)
            # optim_info['rewardGanLoss'].append(reward_gan_loss)
            
            # weighted sum total loss
            # total_loss = \
                    # self._transition_reg_loss_weight * transition_reg_loss + \
                    # self._reward_reg_loss_weight * reward_reg_loss + \
                    # self._transition_gan_loss_weight * transition_gan_loss + \
                    # self._reward_gan_loss_weight * reward_gan_loss + \
                    # done_loss
            # total_loss.backward()

            # transition_loss = self._transition_reg_loss_weight * transition_reg_loss + \
                    # self._transition_gan_loss_weight * transition_gan_loss

            # transition_reg_loss.backward()
            # reward_reg_loss.backward()
            total_loss = transition_reg_loss + reward_reg_loss
            total_loss.backward()
            self._transition_optimizer.step()
            self._reward_optimizer.step()

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

