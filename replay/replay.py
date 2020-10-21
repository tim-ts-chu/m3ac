
import torch

# field name and dimension
BufferFields = {
        'state': 111,
        'action': 8,
        'reward': 1,
        'done': 1,
        'next_state': 111,
        }

class ReplayBuffer:
    '''
    This is a replay buffer for saving step data of trajecries.
    All of the data are saved in a preallocated memory space 
    in a torch.Tensor form. So how large the buffer size can be depends
    on how much memory the machine has.
    '''

    def __init__(self, buffer_size: int, device: int=None):

        self._buffer_size = buffer_size
        self._device = device

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

    def sample(self, batch_size: int) -> torch.Tensor:
        '''
        Sample number of batch size data from replay buffer
        '''
        if batch_size > self._curr_size:
            raise RuntimeError(f'batch_size [{batch_size}] is larger than current data count [{self._curr_size}].')

        indeces = torch.randint(0, self._curr_size, (batch_size,))
        samples = {}
        for field in BufferFields.keys():
            samples[field] = self._buffer[field][indeces, :]

        return samples

    def sample_traj(self, batch_size: int) -> torch.Tensor:
        pass





