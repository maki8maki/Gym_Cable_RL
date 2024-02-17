from collections import deque
import numpy as np


class Buffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size
    
    def append(self, transition):
        raise NotImplementedError
    
    def sample(self, batch_size):
        raise NotImplementedError

    def __repr__(self):
        main_str = f'{self.__class__.__name__}(memory_size={self.memory_size})'
        return main_str

class ReplayBuffer(Buffer):
    '''
    観測・行動は-1 ~ 1に正規化されているものとして扱う
    '''
    def __init__(self, memory_size):
        super().__init__(memory_size)
        self.memory = deque([], maxlen = memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states      = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_states = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        rewards     = np.array([self.memory[index]['reward'] for index in batch_indexes])
        actions     = np.array([self.memory[index]['action'] for index in batch_indexes])
        dones       = np.array([self.memory[index]['done'] for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}
