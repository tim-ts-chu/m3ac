
import torch

class FakeEnv:
    def __init__(self, model_agent):

        self._model_agent = model_agent
        # TODO should handle ensembles

    def step(self, s, a):
        """
        @param s: (batch, state_dim)
        @param a: (batch, action_dim)
        """
        # should handle batch input
        next_s = self._get_obs(s, a)
        reward = self._get_rewared(s, a, next_s.detach())
        done = self._get_done(s, a, next_s)
        info = self._get_info(s, a, next_s)
        return next_s, reward, done, info

    def _get_obs(self, s, a):
        diff = self._model_agent.transition(s, a)
        next_s = s + diff
        return next_s

    def _get_rewared(self, s, a, next_s):
        reward = self._model_agent.reward(s, a, next_s)
        return reward

    def _get_done(self, s, a, next_s):
        batch_size, _ = next_s.shape
        done = torch.zeros(batch_size, 1)
        return done

    def _get_info(self, s, a, next_s):
        return {}

