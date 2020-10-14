
import math
import torch
import logging
from typing import Tuple, Dict, List
from replay import BufferFields
from agent.models import PolicyNetwork, QNetwork
from torch.nn.parallel import DistributedDataParallel as DDP

class Gaussian:
    '''
    Tanh squashed Gaussian Distribution
    '''

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, squash: bool=True):
        self._mean = mean
        min_log_std = -20
        max_log_std = 2
        self._log_std = torch.clamp(log_std,
                min=min_log_std,
                max=max_log_std)
        self._std = torch.exp(self._log_std)
        self._dim = mean.shape[1]
        self._squash = squash

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Draw a sample from Gaussian distribution
        '''

        noise = self._std * torch.normal(torch.zeros_like(self._mean), torch.ones_like(self._mean))
        sample = self._mean + noise

        log_pi = self.loglikelihood(sample)

        if self._squash :
            sample = torch.tanh(sample)

        return sample, log_pi

    def loglikelihood(self, samples: torch.Tensor) -> torch.Tensor:
        '''
        Compute log likelihood of samples
        '''
        EPS = 1e-8
        z = (samples - self._mean) / (self._std + EPS)
        loglikelihood = -(torch.sum(self._log_std + 0.5 * z ** 2, dim=-1, keepdim=True)
                + 0.5 * self._dim * math.log(2 * math.pi))

        # because of squash
        if self._squash:
            loglikelihood -= torch.sum(torch.log((1 - torch.tanh(samples) ** 2) + EPS), dim=-1, keepdim=True)

        return loglikelihood

class SACAgent:
    '''
    This class implemented soft actor critic agent which can be used
    in sac algorithm. It includes a policy network and 2 twin Q network.
    '''
    def __init__(self,
            device_id: int,
            world_size: int,
            policy_hidden_size: List,
            q_hidden_size: List,
            model_path: str=None):

        self._device_id = device_id
        self._world_size = world_size

        self._logger = logging.getLogger()

        self._pi = PolicyNetwork(
                state_size=BufferFields['state'],
                action_size=BufferFields['action'],
                hidden_sizes=policy_hidden_size).to(device_id)

        q_param = {
                'state_size': BufferFields['state'],
                'action_size': BufferFields['action'],
                'hidden_sizes': q_hidden_size}

        self._q1 = QNetwork(**q_param).to(device_id)
        self._q2 = QNetwork(**q_param).to(device_id)
        self._q1_target = QNetwork(**q_param).to(device_id)
        self._q2_target = QNetwork(**q_param).to(device_id)

        if model_path:
            self.load_model(model_path)
            self._logger.info(f'load agent model from {model_path}')

        # wrap models after init and load
        self._pi = self._ddp_wrap(self._pi)
        self._q1 = self._ddp_wrap(self._q1)
        self._q2 = self._ddp_wrap(self._q2)

        self._logger.info(self._pi)
        self._logger.info(self._q1)
        self._logger.info(self._q2)
        self._logger.info(self._q1_target)
        self._logger.info(self._q2_target)

        self._eval_mode = False

    def eval_mode(self, eval_mode: bool) -> None:
        '''
        Set network as evaluation mode.
        Only works for some type of network architecture
        '''
        self._eval_mode = eval_mode
        if eval_mode:
            self._pi.eval()
        else:
            self._pi.train()

    def pi_parameters(self) -> Dict:
        '''
        Return pi net paramenters
        '''
        return self._pi.parameters()

    def q1_parameters(self) -> Dict:
        '''
        Return q1 net parameters
        '''
        return self._q1.parameters()

    def q2_parameters(self) -> Dict:
        '''
        Return q2 net parameters
        '''
        return self._q2.parameters()

    def pi(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Sample an action using policy net and gaussian distribution
        '''
        state = state.to(self._device_id)
        mu, log_std = self._pi(state)
        distribution = Gaussian(mu, log_std)
        action, log_pi = distribution.sample()
        return mu, log_std, action, log_pi

    def q(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Return twin q values
        '''
        state = state.to(self._device_id)
        action = action.to(self._device_id)
        q1 = self._q1(state, action)
        q2 = self._q2(state, action)
        return q1, q2

    def q_target(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Return target twin q values
        '''
        state = state.to(self._device_id)
        action = action.to(self._device_id)
        q1_target = self._q1_target(state, action)
        q2_target = self._q2_target(state, action)
        return q1_target, q2_target

    def update_q_target(self, update_tau: float) -> None:
        '''
        Soft update target q-network parameters
        '''
        q1_state_dict = self._q1.state_dict()
        q2_state_dict = self._q2.state_dict()
        self._soft_update_state_dict(self._q1_target, q1_state_dict, update_tau)
        self._soft_update_state_dict(self._q2_target, q2_state_dict, update_tau)

    def save_model(self, path: str) -> None:
        '''
        Save models without ddp wrapper to the given path
        '''
        param = {}
        param['q1'] = self._strip_ddp_state_dict(self._q1.state_dict())
        param['q2'] = self._strip_ddp_state_dict(self._q2.state_dict())
        param['pi'] = self._strip_ddp_state_dict(self._pi.state_dict())
        torch.save(param, path)

    def load_model(self, path: str) -> None:
        '''
        Load model from given path without ddp wrapper,
        and assuming always load cuda:0 model, so we map
        cuda:0 to current assigned device id.
        '''
        map_location = {'cuda:0': f'cuda:{self._device_id}'}
        param = torch.load(path, map_location=map_location)

        self._q1.load_state_dict(param['q1'])
        self._q2.load_state_dict(param['q2'])
        self._q1_target.load_state_dict(param['q1'])
        self._q2_target.load_state_dict(param['q2'])
        self._pi.load_state_dict(param['pi'])

    def _ddp_wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        '''
        Wrapper network module using DistributedDataParallel object
        if the world size is larger than 1.
        '''
        if self._world_size > 1:
            return DDP(model, device_ids=[self._device_id])
        else:
            return model

    def _soft_update_state_dict(self, model: torch.nn.Module, state_dict: Dict, tau: float=1) -> None:
        '''
        Soft update state dict of model given state dict and tau
        '''
        state_dict = self._strip_ddp_state_dict(state_dict)
        if tau == 1:
            model.load_state_dict(state_dict)
        elif tau > 0:
            update_sd = {k: tau * state_dict[k] + (1 - tau) * v for k, v in model.state_dict().items()}
            model.load_state_dict(update_sd)

    def _strip_ddp_state_dict(self, state_dict: Dict) -> Dict:
        '''
        DistributedDataParallel prepends 'module.' to every key,
        but for the general purpose, we want to save and load
        state dict without prepended key.
        '''
        clean_state_dict = type(state_dict)()    
        for k, v in state_dict.items():    
            key = k[7:] if k[:7] == "module." else k    
            clean_state_dict[key] = v    
        return clean_state_dict

