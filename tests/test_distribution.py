
import sys
sys.path.insert(0,'..')
import math
import torch
import unittest
import numpy as np
from agent.sac_agent import Gaussian

class TestDistribution(unittest.TestCase):

    def test_loglikelihood_squashed(self):
        loc = 0.0
        scale = 1.0
        distribution = Gaussian(torch.tensor([[loc]]), torch.log(torch.tensor([[scale]])), True)
        prob = distribution.loglikelihood(torch.tensor(0, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.3989, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(.5, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.4477, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(-.5, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.4477, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(.9, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.5465, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(-.9, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.5465, abs_tol=1e-4), f'prob={prob}')

    def test_loglikelihood_normal(self):
        loc = 0.0
        scale = 1.0
        distribution = Gaussian(torch.tensor([[loc]]), torch.log(torch.tensor([[scale]])), False)
        prob = distribution.loglikelihood(torch.tensor(0, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.39894, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(1, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.24197, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(-1, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.24197, abs_tol=1e-4), f'prob={prob}')

        loc = np.float32(0)
        scale = np.float32(2.5)
        distribution = Gaussian(torch.tensor([[loc]]), torch.log(torch.tensor([[scale]])), False)
        prob = distribution.loglikelihood(torch.tensor(0, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.15958, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(2.5, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.09679, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(-2.5, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.09679, abs_tol=1e-4), f'prob={prob}')

        loc = np.float32(-1.5)
        scale = np.float32(2.5)
        distribution = Gaussian(torch.tensor([[loc]]), torch.log(torch.tensor([[scale]])), False)
        prob = distribution.loglikelihood(torch.tensor(0, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.13329, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(2.5, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.04437, abs_tol=1e-4), f'prob={prob}')
        prob = distribution.loglikelihood(torch.tensor(-2.5, dtype=torch.float32)).exp()
        self.assertTrue(math.isclose(prob, 0.14731, abs_tol=1e-4), f'prob={prob}')

    def test_sample_normal(self):
        # only test output dimension, doesn't check the number because it's stohastic
        loc = torch.tensor([
            [0., 1.],
            [0., 1.],
            [0., 1.]])
        scale = torch.tensor([
            [1., 2.],
            [1., 2.],
            [1., 2.]])
        distribution = Gaussian(loc, scale)
        sample, log_pi = distribution.sample()
        self.assertEqual(sample.shape, (3, 2))
        self.assertEqual(log_pi.shape, (3, 1))



