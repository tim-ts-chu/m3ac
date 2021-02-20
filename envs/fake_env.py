
import torch

class FakeEnv:
    def __init__(self, real_env_id, model_agent):

        self._real_env_id = real_env_id
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
        if self._real_env_id == 'Walker2d-v3':
            is_healthy = is_walker_healthy(next_s)
            return torch.logical_not(is_healthy).view(-1, 1)
        elif self._real_env_id == 'Hopper-v3':
            is_healthy = is_hopper_healthy(next_s)
            return torch.logical_not(is_healthy).view(-1, 1)
        
        # for other no unhealthy termination environments
        batch_size, _ = next_s.shape
        done = torch.zeros(batch_size, 1)
        return done

    def _get_info(self, s, a, next_s):
        return {}

def is_walker_healthy(state):
    # handle batch dim
    z = state[:, 0]
    angle = state[:, 1]

    min_z, max_z = (0.8, 2.0) # healthy_z_range
    min_angle, max_angle = (-1.0, 1.0) # healthy_angle_range

    healthy_z = torch.logical_and(min_z < z, z < max_z)
    healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
    is_healthy = torch.logical_and(healthy_z, healthy_angle)

    return is_healthy # (batch, )

def is_hopper_healthy(state):
    # handle batch dim
    z = state[:, 0]
    angle = state[:, 1]

    state_excluded = state[2:]

    min_state, max_state = (-100.0, 100.0) # self._healthy_state_range
    min_z, max_z = (0.7, float('inf')) # self._healthy_z_range
    min_angle, max_angle = (-0.2, 0.2) # self._healthy_angle_range

    healthy_state = torch.all(
        np.logical_and(min_state < state_excluded, state_excluded < max_state), dim=1)

    healthy_z = torch.logical_and(min_z < z, z < max_z)
    healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)

    is_healthy = all((healthy_state, healthy_z, healthy_angle))

    return is_healthy

