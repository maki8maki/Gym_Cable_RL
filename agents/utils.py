import numpy as np
import scipy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import cv2

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

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

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

class RL:
    def __init__(self):
        self.info = {}
        self.device = 'cpu'
    
    def batch_to_tensor(self, batch, key_list=['states', 'actions', 'next_states', 'rewards', 'dones']):
        return_list = {}
        for key in key_list:
            if isinstance(batch[key], torch.Tensor):
                item = batch[key]
                if item.dtype != torch.float:
                    item = item.to(torch.float)
                if item.device != self.device:
                    item = item.to(self.device)
            else:
                item = torch.tensor(batch[key], dtype=torch.float, device=self.device)
            return_list[key] = item
        return return_list
    
    def update_from_batch(self, batch):
        raise NotImplementedError
    
    def get_action(self, state, deterministic=False):
        if isinstance(state, torch.Tensor):
            state_tensor = state
            if state_tensor.dtype != torch.float:
                state_tensor = state_tensor.to(torch.float)
            if state_tensor.device != self.device:
                state_tensor = state_tensor.to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        return state_tensor
    
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
    
    def to(self, device):
        self.device = device
    
    def tensor2ndarray(self, t):
        return t.to('cpu').detach().numpy().copy()
    
    def _get_name(self):
        return self.__class__.__name__
        
    def __repr__(self) -> str:
        return self._get_name() +'()'

class MyTrans(nn.Module):
    def __init__(self, img_width, img_height):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
    
    def __call__(self, img):
        """
        resize, shapeの変更, [-1, 1]を[0, 1]に変換
        """
        return cv2.resize(img, (self.img_width, self.img_height)).transpose(2, 0, 1) * 0.5 + 0.5
