
import gym
import torch
import numpy as np
from replay.replay import BufferFields
from gym.envs.registration import registry, register, make, spec

register(                                                                     
    id='Hopper-v4',
    kwargs={'terminate_when_unhealthy': False},
    entry_point='gym.envs.mujoco.hopper_v3:HopperEnv',                         
    max_episode_steps=1000,                                                   
    reward_threshold=3800.0,                                                  
)

register(
    id='HalfCheetah-v4',
    entry_point='gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv',
    max_episode_steps=1000,                               
    reward_threshold=4800.0,
)

register(
    id='Walker2d-v4',
    kwargs={'terminate_when_unhealthy': False},
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco.walker2d_v3:Walker2dEnv',
)

register(
    id='Ant-v4',
    kwargs={'terminate_when_unhealthy': False},
    entry_point='gym.envs.mujoco.ant_v3:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v4',
    kwargs={'terminate_when_unhealthy': False},
    entry_point='gym.envs.mujoco.humanoid_v3:HumanoidEnv',
    max_episode_steps=1000,
)

class GymEnv:

    def __init__(self, *args, **kwargs):
        self._env = gym.make(*args, **kwargs)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def getattr(self, attr_name):
        return getattr(self._env, attr_name)

    def step(self, action: torch.Tensor, run_for_n_step = 1):
        r_sum = 0
        for i in range(run_for_n_step):
            try:
                o, r, d, info = self._env.step(action.numpy())
            except Exception as e:
                print(e)
                o = np.zeros(BufferFields['state'])
                d = True
                r = -10
                info = {}
            r_sum += r
            if d:
                break
        return torch.from_numpy(o).to(torch.float32).view(1, -1), torch.tensor(r_sum).to(torch.float32), torch.tensor(d), info

    def reset(self):
        o = self._env.reset()
        return torch.from_numpy(o).to(torch.float32).view(1, -1)

    def render(self, step_reward=None, cumulative_reward=None):
        return self._env.render('rgb_array') # (500, 500, 3)


