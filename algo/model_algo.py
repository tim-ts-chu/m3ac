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
            reward_reg_loss_weight: float,
            gan_loss_weight: float,
            h_step_loss: int,
            trans_model_lr: float,
            reward_model_lr: float,
            trans_weight_decay: float,
            reward_weight_decay: float,
            num_epoch: int,
            epoch_data_size: int,
            model_batch_size: int):

        self._device_id = device_id
        self._real_buffer = real_buffer
        self._imag_buffer = imag_buffer
        self._model_agent = model_agent
        self._policy_agent = policy_agent
        self._disc_agent = disc_agent
        self._fake_env = fake_env

        self._transition_reg_loss_weight = transition_reg_loss_weight
        self._reward_reg_loss_weight = reward_reg_loss_weight
        self._gan_loss_weight = gan_loss_weight
        self._h_step_loss = h_step_loss
        self._num_epoch = num_epoch
        self._epoch_data_size = epoch_data_size
        self._model_batch_size = model_batch_size

        self._transition_optimizer = torch.optim.AdamW(
                model_agent.transition_params(),
                lr=trans_model_lr,
                weight_decay=trans_weight_decay)

        self._reward_optimizer = torch.optim.AdamW(
                model_agent.reward_params(),
                lr=reward_model_lr,
                weight_decay=reward_weight_decay)

    def optimize_agent(self, step: int) -> Dict:
        optim_info = {}
        optim_info['transitionRegLoss'] = []
        optim_info['rewardRegLoss'] = []
        optim_info['GanLoss'] = []
        optim_info['doneLoss'] = []

        for it in range(self._num_epoch):
            self._transition_optimizer.zero_grad()
            self._reward_optimizer.zero_grad()
            real_samples = self._real_buffer.sample_sequence(self._model_batch_size, self._h_step_loss, self._device_id)

            # mse loss
            next_state, reward, done, _ = self._fake_env.step(real_samples['state'][:,0,:], real_samples['action'][:,0,:])
            transition_reg_loss = F.mse_loss(next_state, real_samples['next_state'][:,0,:])
            reward_reg_loss = F.mse_loss(reward, real_samples['reward'][:,0,:])
            optim_info['transitionRegLoss'].append(transition_reg_loss)
            optim_info['rewardRegLoss'].append(reward_reg_loss)


            # gan loss
            # real_samples = self._real_buffer.sample_sequence(self._model_batch_size, self._h_step_loss, self._device_id)
            # next_state, reward, done, _ = self._fake_env.step(real_samples['state'][:,0,:], real_samples['action'][:,0,:])
            next_state_diff = self._model_agent.transition(real_samples['state'][:,0,:], torch.rand_like(real_samples['action'][:,0,:]) * 2 - 1)
            logits, pred = self._disc_agent.discriminate(
                    real_samples['state'][:,0,:],
                    # real_samples['action'][:,0,:],
                    # real_samples['reward'][:,0,:],
                    # real_samples['done'][:,0,:],
                    next_state_diff)
            true_labels = torch.full((self._model_batch_size, 1), .8, dtype=logits.dtype, device=logits.device) + \
                    torch.randn((self._model_batch_size, 1), dtype=logits.dtype, device=logits.device)/10.0
            gan_loss = F.binary_cross_entropy_with_logits(logits, true_labels)
            optim_info['GanLoss'].append(gan_loss)

            total_loss = self._transition_reg_loss_weight * transition_reg_loss + \
                    self._reward_reg_loss_weight * reward_reg_loss + \
                    self._gan_loss_weight * gan_loss
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

