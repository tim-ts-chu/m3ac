
import sys
sys.path.insert(0,'..')
import torch
from replay import ReplayBuffer, BufferFields
import unittest

class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self._large_buffer_size = 2560 # this number should be larger than testing data size
        self._small_buffer_size = 100 # this number should be smaller than testing data size
        self._data_size = 1000
        self._testing_data = []

        for i in range(self._data_size):
            sample = {}
            for k, dim in BufferFields.items():
                sample[k] = torch.rand((1, dim))
            self._testing_data.append(sample)

    def test_push(self):
        replay_buffer = ReplayBuffer(self._large_buffer_size)
        for i in range(self._data_size):
            replay_buffer.push(**self._testing_data[i])
            self.assertEqual(replay_buffer.size, i+1)

    def test_push_and_check(self):
        replay_buffer = ReplayBuffer(self._large_buffer_size)
        for i in range(self._data_size):
            replay_buffer.push(**self._testing_data[i])
            self.assertEqual(replay_buffer.size, i+1)

        for i in range(self._data_size):
            for key in BufferFields.keys():
                buffer_data = replay_buffer._buffer[key][i,:].view(1, -1)
                testing_data = self._testing_data[i][key]
                self.assertTrue(torch.equal(buffer_data, testing_data))

    def test_sampling_large_buffer(self):
        replay_buffer = ReplayBuffer(self._large_buffer_size)
        for i in range(self._data_size):
            replay_buffer.push(**self._testing_data[i])
            self.assertEqual(replay_buffer.size, i+1)

        for i in range(100): # test for 100 times
            batch_size = 16 
            samples = replay_buffer.sample(batch_size)
            for idx in range(batch_size):
                data_idx = None
                key = list(BufferFields.keys())[0]
                for j in range(self._data_size):
                    if torch.equal(samples[key][idx,:].view(1, -1), self._testing_data[j][key]):
                        data_idx = j
                        break
                else:
                    self.fail('sampled data is not in the testing data')

                # data for all key are matched
                for key in BufferFields.keys():
                    self.assertTrue(torch.equal(samples[key][idx,:].view(1, -1), self._testing_data[data_idx][key]))

    def test_circling(self):
        replay_buffer = ReplayBuffer(self._small_buffer_size)
        for i in range(self._data_size):
            replay_buffer.push(**self._testing_data[i])

        for i in range(self._small_buffer_size):
            for key in BufferFields.keys():
                data_idx = self._data_size - self._small_buffer_size + i
                buffer_data = replay_buffer._buffer[key][i,:].view(1, -1)
                testing_data = self._testing_data[data_idx][key]
                self.assertTrue(torch.equal(buffer_data, testing_data))


if __name__ == '__main__':
    unittest.main()




