
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

    def step(self, action: torch.Tensor, run_for_n_step = 1):
        r_sum = 0
        for i in range(run_for_n_step):
            o, r, d, info = self._env.step(action.numpy())
            r_sum += r
            if d:
                break
        return torch.from_numpy(o).to(torch.float32).view(1, -1), torch.tensor(r_sum).to(torch.float32), torch.tensor(d), info

    def reset(self):
        o = self._env.reset()
        return torch.from_numpy(o).to(torch.float32).view(1, -1)

    def render(self, step_reward=None, cumulative_reward=None):
        return self._env.render('rgb_array') # (500, 500, 3)


