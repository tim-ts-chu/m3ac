
import sys
sys.path.insert(0,'..')
import torch
import unittest
from agent.models import QNetwork, PolicyNetwork

class TestDistribution(unittest.TestCase):

    def test_model(self):

        batch_size = 256
        state_size = 2000
        action_size = 2
        s = torch.normal(torch.zeros(batch_size, state_size))
        a = torch.normal(torch.zeros(batch_size, action_size))

        p_net = PolicyNetwork(state_size, action_size, [256, 128])
        mu, logstd = p_net(s)
        self.assertTrue(mu.shape, (batch_size, action_size))

        q_net = QNetwork(state_size, action_size, [256, 128])
        values = q_net(s, a)
        self.assertTrue(values.shape, (batch_size, 1))


