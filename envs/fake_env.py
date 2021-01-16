
import torch

def get_fake_env(env_id, model_agent, real_env):
    # TODO remove parameters
    return BaseFakeEnv(model_agent, real_env)

class BaseFakeEnv:
    def __init__(self, model_agent, real_env):

        self._real_env = real_env # we will need some real env parameters
        self._transition = model_agent.transition
        self._reward = model_agent.reward
        self._done = model_agent.done

    def step(self, s, a):
        """
        @param s: (batch, state_dim)
        @param a: (batch, action_dim)
        """
        # should handle batch input
        next_s = self._get_obs(s, a)
        reward = self._get_rewared(s, a, next_s)
        done = self._get_done(s, a, next_s)
        info = self._get_info(s, a, next_s)
        return next_s, reward, done, info

    def _get_obs(self, s, a):
        diff = self._transition(s, a).rsample()
        next_s = s + diff
        return next_s

    def _get_rewared(self, s, a, next_s):
        reward = self._reward(s, a, next_s).rsample()
        return reward

    def _get_done(self, s, a, next_s):
        # TODO remove prdiction part (shouldn't be necessary)
        if self._done:
            _, done = self._done(s, a, next_s)
        else:
            batch_size, _ = next_s.shape
            done = torch.zeros(batch_size).bool()
        return done

    def _get_info(self, s, a, next_s):
        return {}

