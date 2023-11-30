from collections import deque
import numpy as np
import torch.nn as nn

def size_after_conv(h, ksize, stride=1, padding=0):
    return ((h - ksize + 2 * padding) // stride) + 1

def size_after_pooling(h, ksize):
    return h // ksize

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self,x):
        return x.reshape(self.shape)

class ReplayBuffer:
    def __init__(self, memory_size) -> None:
        self.memory_size = memory_size
        self.memory = deque([], maxlen = memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states      = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_states = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        rewards     = np.array([self.memory[index]['reward'] for index in batch_indexes])
        actions     = np.array([self.memory[index]['action'] for index in batch_indexes])
        successes   = np.array([self.memory[index]['success'] for index in batch_indexes])
        dones       = np.array([self.memory[index]['done'] for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'successes': successes, 'dones': dones}