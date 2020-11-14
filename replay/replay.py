
import torch

# field name and dimension
BufferFields = {
        'state': None,
        'action': None,
        'reward': None,
        'done': None,
        'next_state': None,
        }

def set_buffer_dim(state_dim, action_dim, reward_dim, done_dim):
    BufferFields['state'] = state_dim
    BufferFields['action'] = action_dim
    BufferFields['reward'] = reward_dim
    BufferFields['done'] = done_dim
    BufferFields['next_state'] = state_dim

class ReplayBuffer:
    '''
    This is a replay buffer for saving step data of trajecries.
    All of the data are saved in a preallocated memory space 
    in a torch.Tensor form. So how large the buffer size can be depends
    on how much memory the machine has.
    '''

    def __init__(self, buffer_size: int, dtype: torch.dtype=None, device: int=None):

        self._buffer_size = buffer_size
        self._device = device
        self._dtype = dtype

        self._buffer = {}
        self._curr_size = 0
        self._write_idx = 0

        # initialize buffer for each field
        for name in BufferFields.keys():
            field_dim = BufferFields[name]
            self._buffer[name] = torch.empty((self._buffer_size, field_dim), device=self._device, requires_grad=False)

    @property
    def size(self) -> int:
        return self._curr_size

    def push(self, **kwargs) -> None:
        '''
        Push a data record into replay buffer
        '''
        if kwargs.keys() != BufferFields.keys():
            raise RuntimeError(f'push data into an unexisting field: {kwargs.keys()}!={BufferFields.keys()}')

        for field in BufferFields.keys():
            self._buffer[field][self._write_idx, :] = kwargs[field]

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

    def sample(self, batch_size: int, device_id=None) -> torch.Tensor:
        '''
        Sample number of batch size data from replay buffer
        '''
        if batch_size > self._curr_size:
            raise RuntimeError(f'batch_size [{batch_size}] is larger than current data count [{self._curr_size}].')

        indeces = torch.randint(0, self._curr_size, (batch_size,))
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

        # fill buffer by done flag
        buffer_indeces = torch.arange(batch_size)
        sample_indeces = torch.randint(0, self._curr_size, (batch_size,))
        for t in range(max_steps):
            for field in BufferFields.keys():
                samples[field][buffer_indeces, t, :] = self._buffer[field][sample_indeces,:].to(device)

            # handle terminated sequence
            done_mask = (samples['done'][buffer_indeces,t,:] < 0.5).view(-1) # done is saved as 1. or 0. in float

            if done_mask.sum() < 1:
                break

            buffer_indeces = buffer_indeces[done_mask]
            sample_indeces = sample_indeces[done_mask] + 1
            sample_indeces[sample_indeces>=self._buffer_size] -= self._buffer_size # circling if need

        return samples











