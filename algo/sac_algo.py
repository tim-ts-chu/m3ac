
import torch
import numpy as np
from typing import Dict, Tuple
from replay.replay import BufferFields, ReplayBuffer
from agent.sac_agent import SACAgent, Gaussian
from algo.model_algo import ModelAlgorithm

class SACAlgorithm:
    '''
    This class implemented the soft actor critic algorithm.
    It needs to coperate with sac agent
    '''
    def __init__(self,
            device_id: int,
            agent: SACAgent,
            model_algo: ModelAlgorithm,
            real_buffer: ReplayBuffer,
            num_updates: int,
            real_batch_size: int,
            imag_batch_size: int,
            discount: float=0.99,
            clip_grad_norm: float=1e9,
            learning_rate: float=3e-4,
            update_target_interval: int=1,
            update_target_tau: float=0.005,
            target_entropy=None):

        self._device_id = device_id
        self._agent = agent
        self._model_algo = model_algo
        self._real_buffer = real_buffer
        self._num_updates = num_updates
        self._real_batch_size = real_batch_size
        self._imag_batch_size = imag_batch_size

        self._discount = discount
        self._clip_grad_norm = clip_grad_norm
        self._lr = learning_rate
        self._update_target_interval = update_target_interval
        self._update_target_tau = update_target_tau

        # for agent
        self._pi_optimizer = torch.optim.Adam(self._agent.pi_parameters(), lr=self._lr)
        self._q1_optimizer = torch.optim.Adam(self._agent.q1_parameters(), lr=self._lr)
        self._q2_optimizer = torch.optim.Adam(self._agent.q2_parameters(), lr=self._lr)

        # for alpha
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device_id)
        self._log_alpha_optimizer = torch.optim.Adam((self._log_alpha,), lr=self._lr)
        if target_entropy:
            self._target_entropy = target_entropy
        else:
            self._target_entropy = -np.prod(BufferFields['action'])

    @property
    def discount(self) -> float:
        return self._discount

    @property
    def alpha(self) -> torch.Tensor:
        return self._log_alpha.exp().detach().to(self._device_id)

    def optimize_agent(self, step: int) -> Dict:
        '''
        Optimize agent for one step given a batch of samples
        '''

        # obtain samples
        if self._real_batch_size > 0 and self._imag_batch_size > 0:
            samples_real = self._real_buffer.sample(self._real_batch_size)
            samples_imag = self._model_algo.generate_samples(self._imag_batch_size)
            samples = {}
            for k, v in samples_imag.items():
                samples[k] = torch.cat((samples_real[k].to(self._device_id), samples_imag[k].to(self._device_id)), dim=0)
        elif self._real_batch_size > 0:
            samples = self._real_buffer.sample(self._real_batch_size)
        elif self._imag_batch_size > 0:
            samples = self._model_algo.generate_samples(self._imag_batch_size)
        else:
            raise Exception('At least one type of batch size should > 0')

        optim_info = {}
        optim_info['alpha'] = []
        optim_info['q1Loss'] = []
        optim_info['q2Loss'] = []
        optim_info['piLoss'] = []
        optim_info['piGradNorm'] = []
        optim_info['q1GradNorm'] = []
        optim_info['q2GradNorm'] = []

        for i in range(self._num_updates):

            q1_loss, q2_loss, pi_loss, alpha_loss = self._loss(samples)
            optim_info['alpha'].append(self.alpha)
            optim_info['q1Loss'].append(q1_loss)
            optim_info['q2Loss'].append(q2_loss)
            optim_info['piLoss'].append(pi_loss)

            # optimize alpha
            self._log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._log_alpha_optimizer.step()

            # optimize pi
            self._pi_optimizer.zero_grad()
            pi_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._agent.pi_parameters(), self._clip_grad_norm)
            self._pi_optimizer.step()
            optim_info['piGradNorm'].append(grad_norm)

            # optimize q
            self._q1_optimizer.zero_grad()
            q1_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._agent.q1_parameters(), self._clip_grad_norm)
            self._q1_optimizer.step()
            optim_info['q1GradNorm'].append(grad_norm)

            self._q2_optimizer.zero_grad()
            q2_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._agent.q2_parameters(), self._clip_grad_norm)
            self._q2_optimizer.step()
            optim_info['q2GradNorm'].append(grad_norm)

            # update target q
            if step % self._update_target_interval == 0:
                self._agent.update_q_target(self._update_target_tau)

        return optim_info

    def _loss(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Compute optimization loss given samples
        '''
        for k, v in samples.items():
            samples[k] = v.to(self._device_id)

        # q loss
        q1, q2 = self._agent.q(samples['state'], samples['action'])

        with torch.no_grad():
            _, _, next_action, next_log_pi = self._agent.pi(samples['next_state'])
            q1_target, q2_target = self._agent.q_target(samples['next_state'], next_action)

            q_target = torch.min(q1_target, q2_target)
            target_value = (q_target - self.alpha * next_log_pi).detach()

            y = samples['reward'] + (1 - samples['done'].float()) * self._discount * target_value

        q1_loss = torch.nn.functional.mse_loss(q1, y)
        q2_loss = torch.nn.functional.mse_loss(q2, y)

        # pi loss
        mu, log_std, action, log_pi = self._agent.pi(samples['state'])
        q1_pi , q2_pi = self._agent.q(samples['state'], action)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_losses = self.alpha * log_pi - q_pi
        pi_loss = pi_losses.mean()

        # alpha loss
        alpha_losses = - self._log_alpha * (log_pi.detach() + self._target_entropy)
        alpha_loss = alpha_losses.mean()

        return q1_loss, q2_loss, pi_loss, alpha_loss


