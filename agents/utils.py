from collections import deque
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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

class SSIMLoss(nn.Module):
    """Copy from https://zenn.dev/taikiinoue45/articles/bf7d2314ab4d10"""

    def __init__(self, channel: int = 3, kernel_size: int = 11, sigma: float = 1.5) -> None:
        """Computes the structural similarity (SSIM) index map between two images.

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.channel, self.kernel_size, self.sigma)

    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, channel: int, kernel_size: int, sigma: float) -> Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(channel, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d

class ReplayBuffer:
    def __init__(self, memory_size):
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
        dones       = np.array([self.memory[index]['done'] for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}

class RL:
    def __init__(self):
        pass
    
    def batch_to_tensor(self, batch, key_list=['states', 'actions', 'next_states', 'rewards', 'dones']):
        return_list = []
        for key in key_list:
            if isinstance(batch[key], torch.Tensor):
                item = batch[key]
                if item.dtype != torch.float:
                    item = item.to(torch.float)
                if item.device != self.device:
                    item = item.to(self.device)
            else:
                item = torch.tensor(batch[key], dtype=torch.float, device=self.device)
            return_list.append(item)
        return return_list
    
    def update_from_batch(self, batch):
        raise NotImplementedError
    
    def get_action(self, state, deterministic=False):
        raise NotImplementedError
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def state_dict(self):
        raise NotImplementedError()
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError()
    
    def eval(self):
        raise NotImplementedError()
    
    def train(self):
        raise NotImplementedError()