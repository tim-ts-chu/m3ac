
import sys
sys.path.insert(0,'..')
import torch
from replay.replay import ReplayBuffer, BufferFields, set_buffer_dim
import unittest

class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self._large_buffer_size = 2560 # this number should be larger than testing data size
        self._small_buffer_size = 100 # this number should be smaller than testing data size
        self._data_size = 1000
        self._testing_data = []

        set_buffer_dim(5, 3, 1, 1)

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

    def deprecated_test_push_batch(self):
        sample = {}
        batch_size = 60
        for k, dim in BufferFields.items():
            sample[k] = torch.rand((batch_size, dim))

        replay_buffer = ReplayBuffer(self._small_buffer_size)

        replay_buffer.push_batch(**sample)
        for i in range(batch_size):
            for key in BufferFields.keys():
                buffer_data = replay_buffer._buffer[key][i,:]#.view(1, -1)
                self.assertTrue(torch.equal(buffer_data, sample[key][i,:]))

    def deprecated_test_push_batch_circling(self):
        sample = {}
        batch_size = 77 
        for k, dim in BufferFields.items():
            sample[k] = torch.rand((batch_size, dim))

        replay_buffer = ReplayBuffer(self._small_buffer_size)
        replay_buffer.push_batch(**sample)
        replay_buffer.push_batch(**sample)
        for i in range(54):
            for key in BufferFields.keys():
                buffer_data = replay_buffer._buffer[key][i,:]#.view(1, -1)
                self.assertTrue(torch.equal(buffer_data, sample[key][23+i,:]))

        self.assertEqual(replay_buffer._curr_size, self._small_buffer_size)
        self.assertEqual(replay_buffer._write_idx, 54)


    def test_sample_sequence(self):
        replay_buffer = ReplayBuffer(self._large_buffer_size)

        # create sequence of data and push
        data_seq_len = [1233, 1043, 542, 231, 120, 53]
        #data_seq_len = [2, 3, 4, 5]
        testing_data = []
        for seq_len in data_seq_len:
            # print('seq_len', seq_len)
            for e in range(seq_len):
                sample = {}
                for k, dim in BufferFields.items():
                    if k == 'done':
                        if e == seq_len - 1:
                            sample[k] = torch.ones((1, dim)) # done
                        else:
                            sample[k] = torch.zeros((1, dim)) # not done
                    else:
                        sample[k] = torch.rand((1, dim))
                # print('idx', e, 'sample:', sample)
                testing_data.append(sample)
                replay_buffer.push(**sample)


        max_steps = 58
        batch_size = 17 
        sample_seq = replay_buffer.sample_sequence(batch_size, max_steps) #(b, t, d)

        # check dimension
        for key in BufferFields.keys():
            self.assertEqual(sample_seq[key].shape, (batch_size, max_steps, BufferFields[key]))

        # only check first one for now
        for batch_idx in range(batch_size):
            data_idx = None
            key = list(BufferFields.keys())[0]
            for j in range(sum(data_seq_len)):
                if torch.equal(sample_seq[key][batch_idx, 0, :].view(1, -1), testing_data[j][key]):
                    data_idx = j
                    #print('data_idx:', data_idx)
                    break
            else:
                self.fail('sampled data is not in the testing data')

            # data for all key are matched
            for key in BufferFields.keys():
                self.assertTrue(torch.equal(sample_seq[key][batch_idx, 0, :].view(1, -1), testing_data[data_idx][key]))

            # calculated gt remain length and check
            len_sum = 0
            for seq_len in data_seq_len:
                if len_sum > data_idx:
                    break
                len_sum += seq_len
            remain_len = len_sum - data_idx
            done_pos = sample_seq['done'][batch_idx, :, :].argmax()
            if remain_len > max_steps:
                self.assertTrue(torch.equal(sample_seq['done'][batch_idx, :, :], torch.zeros(max_steps, BufferFields['done'])))
            else:
                self.assertEqual(done_pos+1, remain_len)

            # check data
            for key in BufferFields.keys():
                for t in range(max_steps):
                    if t < remain_len:
                        # check before data
                        self.assertTrue(torch.equal(sample_seq[key][batch_idx, t, :], testing_data[data_idx+t][key].view(-1)))
                    else:
                        # check remain zeros
                        self.assertTrue(torch.equal(sample_seq[key][batch_idx, t:, :], torch.zeros(max_steps-remain_len, BufferFields[key])))
                        break


if __name__ == '__main__':
    unittest.main()




