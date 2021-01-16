
import torch

# field name and dimension
BufferFields = {
        'state': None,
        'action': None,
        'reward': None,
        'done': None,
        'next_state': None,
        'end': None,
        }

def set_buffer_dim(state_dim, action_dim):
    BufferFields['state'] = state_dim
    BufferFields['action'] = action_dim
    BufferFields['reward'] = 1
    BufferFields['done'] = 1
    BufferFields['next_state'] = state_dim
    BufferFields['end'] = 1

class ReplayBuffer:
    '''
    This is a replay buffer for saving step data of trajecries.
    All of the data are saved in a preallocated memory space 
    in a torch.Tensor form. So how large the buffer size can be depends
    on how much memory the machine has.
    '''

    def __init__(self, buffer_size: int, dtype: torch.dtype=None, device: torch.device=None):

        self._buffer_size = buffer_size
        self._device = device
        self._dtype = dtype

        self._buffer = {}
        self._curr_size = 0
        self._write_idx = 0
        self._writing = False
        self._writing_range = [0, 0] # (from, to)

        # initialize buffer for each field
        for name in BufferFields.keys():
            field_dim = BufferFields[name]
            self._buffer[name] = torch.empty(
                    (self._buffer_size, field_dim),
                    device=self._device,
                    dtype=self._dtype,
                    requires_grad=False)

    @property
    def size(self) -> int:
        return self._curr_size

    def push(self, seq_end: bool, **kwargs) -> None:
        '''
        Push a data record into replay buffer
        @param new_seq indicator for starting a new sequence
        '''

        if kwargs.keys() != BufferFields.keys():
            raise RuntimeError(f'push data into an unexisting field: {kwargs.keys()}!={BufferFields.keys()}')

        for field in BufferFields.keys():
            self._buffer[field][self._write_idx, :] = kwargs[field]

        if not self._writing:
            # new sequence
            self._writing_range = [self._write_idx, self._write_idx] # (from, to)
        else:
            if self._write_idx == self._writing_range[0]:
                raise RuntimeError('Writing a sequence longer than the buffer size')
            self._writing_range[1] = self._write_idx

        if seq_end:
            self._writing = False
        else:
            self._writing = True

        self._write_idx += 1
        if self._write_idx >= self._buffer_size:
            self._write_idx = 0 # rewrite from begining (oldest data)

        if self._curr_size < self._buffer_size:
            self._curr_size += 1

    def push_batch(self, **kwargs) -> None:
        '''
        Push batch data record into replay buffer
        '''

        raise RuntimeError(f'this function is deprecated due to support for sequence sampling')
        if kwargs.keys() != BufferFields.keys():
            raise RuntimeError(f'push data into an unexisting field: {kwargs.keys()}!={BufferFields.keys()}')
        
        batch_size, _ = kwargs[list(BufferFields.keys())[0]].shape

        if self._buffer_size - self._write_idx >= batch_size:
            # no need to circling 
            for field in BufferFields.keys():
                self._buffer[field][self._write_idx:self._write_idx+batch_size, :] = kwargs[field]

            self._write_idx += batch_size 
            if self._curr_size < self._buffer_size:
                self._curr_size += batch_size
        else:
            # need circling
            remain_idx = self._write_idx + batch_size - self._buffer_size
            for field in BufferFields.keys():
                self._buffer[field][self._write_idx:, :] = kwargs[field][:-remain_idx,:]
                self._buffer[field][:remain_idx, :] = kwargs[field][-remain_idx:,:]

            self._write_idx = remain_idx
            self._curr_size = self._buffer_size

    def sample(self, batch_size: int, device_id=None, index_only=False) -> torch.Tensor:
        '''
        Sample number of batch size data from replay buffer
        '''
        if batch_size > self._curr_size:
            raise RuntimeError(f'batch_size [{batch_size}] is larger than current data count [{self._curr_size}].')

        if self._writing:
            if self._writing_range[0] <= self._writing_range[1]:
                # writing is not circling
                seq_len = self._writing_range[1] - self._writing_range[0] + 1
                low = 0
                high = self._curr_size - seq_len
                indeces = torch.randint(low, high, (batch_size,))
                indeces[indeces>=self._writing_range[0]] += seq_len
            else:
                # writing is circling
                low = self._writing_range[1] + 1
                high = self._writing_range[0]
                indeces = torch.randint(low, high, (batch_size,))
        else:
            indeces = torch.randint(0, self._curr_size, (batch_size,))

        if index_only:
            return indeces

        samples = {}
        for field in BufferFields.keys():
            if device_id is not None:
                samples[field] = self._buffer[field][indeces, :].detach().to(device_id)
            else:
                samples[field] = self._buffer[field][indeces, :].detach()

        return samples

    def sample_sequence(self, batch_size: int, max_steps: int=15, device_id=None):
        '''
        Sample number of batch size sequence, each sequence has length of max_steps.
        If the episode is smaller than max_steps, zeros will be filled in the rest
        of the field.

        @ret tensors with shape (b, t, d) for each field
        '''

        if max_steps > self._curr_size:
            raise RuntimeError(f'max_steps {max_steps} is larger than buffer_size {self._curr_size}')

        # initicalize buffer size
        samples = {}
        if device_id is not None:
            device = device_id
        else:
            device = self._device

        for field in BufferFields.keys():
            samples[field] = torch.zeros(batch_size, max_steps, BufferFields[field], dtype=self._dtype, device=device)

        # fill buffer by sequence according to end flag
        buffer_indeces = torch.arange(batch_size)
        sample_indeces = self.sample(batch_size, index_only=True)
        for t in range(max_steps):
            for field in BufferFields.keys():
                samples[field][buffer_indeces, t, :] = self._buffer[field][sample_indeces,:].to(device)

            # handle terminated sequence
            valid_mask = (samples['end'][buffer_indeces,t,:] < 0.5).view(-1) # end is saved as 1. or 0. in float

            if valid_mask.sum() < 1:
                # short cut for all sequences have ended
                break

            buffer_indeces = buffer_indeces[valid_mask]
            sample_indeces = sample_indeces[valid_mask] + 1
            sample_indeces[sample_indeces>=self._buffer_size] -= self._buffer_size # circling if need

        return samples

