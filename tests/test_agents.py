
import sys
sys.path.insert(0,'..')
import torch
import unittest

from agent.imagine_agent import ImagineAgent

from replay.replay import BufferFields

class TestImagineAgent(unittest.TestCase):

    def test_imagine_agent(self):

        agent = ImagineAgent()

        batch_time = 10
        batch_size = 20
        feature_size = 30

        samples = {}
        for k in BufferFields.keys():
            samples[k] = torch.rand(batch_time, batch_size, BufferFields[k])

        for _ in range(100):
            agent.optimize_agent(samples)


if __name__ == '__main__':

    agent = ImagineAgent()

    batch_time = 10
    batch_size = 20
    feature_size = 30

    samples = {}
    for k in BufferFields.keys():
        samples[k] = torch.rand(batch_time, batch_size, BufferFields[k])


