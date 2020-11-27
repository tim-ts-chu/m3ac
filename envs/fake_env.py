
import torch

def get_fake_env(env_id, model_agent, real_env):
    register = {
            'Hopper-v2': HopperV2FakeEnv,
            'Walker2d-v2': Walker2dV2FakeEnv,
            'HalfCheetah-v2': HalfCheetahV2FakeEnv,
            'Ant-v2': AntV2FakeEnv}

    if env_id not in register.keys():
        raise NotImplementedError(f"{env_id} hasn't been registered")
    else:
        return register[env_id](model_agent, real_env)

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
        raise NotImplementedError("This function haven't been implemented")
        
    def _get_rewared(self, s, a, next_s):
        raise NotImplementedError("This function haven't been implemented")

    def _get_done(self, s, a, next_s):
        raise NotImplementedError("This function haven't been implemented")

    def _get_info(self, s, a, next_s):
        return {}

class HopperV2FakeEnv(BaseFakeEnv):
    """
    Follow this source to implement ground truth reward and done function
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper.py
    """

    def __init__(self, model_agent, real_env):
        super().__init__(model_agent, real_env)

    def _get_obs(self, s, a):
        # assume that transition model return a distribution for state diff
        diff = self._transition(s, a).rsample()
        next_s = torch.clamp(s + diff, -10, 10)
        return next_s

    def _get_rewared(self, s, a, next_s):
        if self._reward:
            reward = self._reward(s, a, next_s).rsample()
        else:
            raise NotImplementedError('Not implemented yet')
            # this is wrong implement (information is not enough)
            # posbefore = s[:, 0]
            # posafter = next_s[:, 0]
            # alive_bonus = 1.0
            # dt = self._real_env.getattr('dt')
            # reward = (posafter - posbefore) / dt
            # reward += alive_bonus
            # action_cost = a.square().sum(dim=1)
            # reward -= 1e-3 * action_cost

        return reward.view(-1, 1)

    def _get_done(self, s, a, next_s):
        if self._done:
            _, done = self._done(s, a, next_s)
        else:
            batch_size, _ = next_s.shape
            done = torch.zeros(batch_size).bool()

            # inf_mask = torch.isinf(next_s).any(dim=1)
            # done[inf_mask] = True

            # abs_mask = (next_s[:, 1:] >= 100).any(dim=1)
            # done[abs_mask] = True

            # height, ang = next_s[:, 0], next_s[:, 1]
            # height_mask = (height <= .7)
            # done[height_mask] = True
            # ang_mask = (ang >= .2)
            # done[ang_mask] = True

        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    # (height > .7) and (abs(ang) < .2))

        return done.view(-1, 1)

class HalfCheetahV2FakeEnv(BaseFakeEnv):
    """
    Follow this source to implement ground truth reward and done function
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
    """

    def __init__(self, model_agent, real_env):
        super().__init__(model_agent, real_env)

    def _get_obs(self, s, a):
        # assume that transition model return a distribution for state diff
        diff = self._transition(s, a).rsample()
        next_s = s + diff
        return next_s

    def _get_rewared(self, s, a, next_s):
        if self._reward:
            reward = self._reward(s, a, next_s).rsample()
        else:
            raise NotImplementedError('Not implemented yet')
            # this is wrong implement (information is not enough)
            # posbefore = s[:, 0]
            # posafter = next_s[:, 0]
            # alive_bonus = 1.0
            # dt = self._real_env.getattr('dt')
            # reward = (posafter - posbefore) / dt
            # reward += alive_bonus
            # action_cost = a.square().sum(dim=1)
            # reward -= 1e-3 * action_cost

        return reward.view(-1, 1)

    def _get_done(self, s, a, next_s):
        if self._done:
            _, done = self._done(s, a, next_s)
        else:
            batch_size, _ = next_s.shape
            done = torch.zeros(batch_size).bool()

            # inf_mask = torch.isinf(next_s).any(dim=1)
            # done[inf_mask] = True

            # abs_mask = (next_s[:, 1:] >= 100).any(dim=1)
            # done[abs_mask] = True

            # height, ang = next_s[:, 0], next_s[:, 1]
            # height_mask = (height <= .7)
            # done[height_mask] = True
            # ang_mask = (ang >= .2)
            # done[ang_mask] = True

        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    # (height > .7) and (abs(ang) < .2))

        return done.view(-1, 1)

class Walker2dV2FakeEnv(BaseFakeEnv):
    def __init__(self, model_agent, real_env):
        super().__init__(model_agent, real_env)

    def _get_obs(self, s, a):
        # assume that transition model return a distribution for state diff
        diff = self._transition(s, a).rsample()
        next_s = s + diff
        return next_s

    def _get_rewared(self, s, a, next_s):
        if self._reward:
            reward = self._reward(s, a, next_s).rsample()
        else:
            raise NotImplementedError('Not implemented yet')
            # this is wrong implement (information is not enough)
            # posbefore = s[:, 0]
            # posafter = next_s[:, 0]
            # alive_bonus = 1.0
            # dt = self._real_env.getattr('dt')
            # reward = (posafter - posbefore) / dt
            # reward += alive_bonus
            # action_cost = a.square().sum(dim=1)
            # reward -= 1e-3 * action_cost

        return reward.view(-1, 1)

    def _get_done(self, s, a, next_s):
        if self._done:
            _, done = self._done(s, a, next_s)
        else:
            batch_size, _ = next_s.shape
            done = torch.zeros(batch_size).bool()

            # inf_mask = torch.isinf(next_s).any(dim=1)
            # done[inf_mask] = True

            # abs_mask = (next_s[:, 1:] >= 100).any(dim=1)
            # done[abs_mask] = True

            # height, ang = next_s[:, 0], next_s[:, 1]
            # height_mask = (height <= .7)
            # done[height_mask] = True
            # ang_mask = (ang >= .2)
            # done[ang_mask] = True

        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    # (height > .7) and (abs(ang) < .2))

        return done.view(-1, 1)

class AntV2FakeEnv(BaseFakeEnv):
    def __init__(self, model_agent, real_env):
        super().__init__(model_agent, real_env)

    def _get_obs(self, s, a):
        # assume that transition model return a distribution for state diff
        diff = self._transition(s, a).rsample()
        next_s = s + diff
        return next_s

    def _get_rewared(self, s, a, next_s):
        if self._reward:
            reward = self._reward(s, a, next_s).rsample()
        else:
            raise NotImplementedError('Not implemented yet')
            # this is wrong implement (information is not enough)
            # posbefore = s[:, 0]
            # posafter = next_s[:, 0]
            # alive_bonus = 1.0
            # dt = self._real_env.getattr('dt')
            # reward = (posafter - posbefore) / dt
            # reward += alive_bonus
            # action_cost = a.square().sum(dim=1)
            # reward -= 1e-3 * action_cost

        return reward.view(-1, 1)

    def _get_done(self, s, a, next_s):
        if self._done:
            _, done = self._done(s, a, next_s)
        else:
            batch_size, _ = next_s.shape
            done = torch.zeros(batch_size).bool()

            # inf_mask = torch.isinf(next_s).any(dim=1)
            # done[inf_mask] = True

            # abs_mask = (next_s[:, 1:] >= 100).any(dim=1)
            # done[abs_mask] = True

            # height, ang = next_s[:, 0], next_s[:, 1]
            # height_mask = (height <= .7)
            # done[height_mask] = True
            # ang_mask = (ang >= .2)
            # done[ang_mask] = True

        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    # (height > .7) and (abs(ang) < .2))

        return done.view(-1, 1)
