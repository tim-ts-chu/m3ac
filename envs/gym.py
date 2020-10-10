
import gym
import torch

class GymEnv:

    def __init__(self, *args, **kwargs):
        self._env = gym.make(*args, **kwargs)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def step(self, action: torch.Tensor):
        o, r, d, info = self._env.step(action.numpy())
        return torch.from_numpy(o), torch.tensor(r), torch.tensor(d), info

    def reset(self):
        o = self._env.reset()
        return torch.from_numpy(o)

    def render(self, step_reward=None, cumulative_reward=None):
        return self._env.render('rgb_array') # (500, 500, 3)


