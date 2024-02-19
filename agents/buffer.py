from collections import deque
import numpy as np
import torch

from .utils import discount_cumsum

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

class PPOBuffer(Buffer):
    def __init__(self, memory_size, obs_dim, act_dim, gamma=0.99, lam=0.95):
        super().__init__(memory_size)
        self.obs = np.zeros((memory_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((memory_size, act_dim), dtype=np.float32)
        self.adv = np.zeros(memory_size, dtype=np.float32)
        self.rew = np.zeros(memory_size, dtype=np.float32)
        self.ret = np.zeros(memory_size, dtype=np.float32)
        self.val = np.zeros(memory_size, dtype=np.float32)
        self.logp = np.zeros(memory_size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx = 0, 0
    
    def append(self, obs, act, rew, val, logp):
        assert self.ptr < self.memory_size
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew[path_slice], last_val)
        vals = np.append(self.val[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[:-1] - vals[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)
        self.ret[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def sample(self):
        assert self.ptr == self.memory_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean = np.mean(self.adv)
        adv_diff = self.adv - adv_mean
        adv_std = np.mean(adv_diff**2)
        self.adv = adv_diff / adv_std
        return {'obs': self.obs, 'act': self.act, 'ret': self.ret, 'adv': self.adv, 'logp': self.logp}
