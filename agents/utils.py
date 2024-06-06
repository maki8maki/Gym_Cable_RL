import dataclasses
import random
from typing import Tuple, Union

import cv2
import numpy as np
import scipy
import torch as th
import torch.nn as nn
import torch.nn.functional as NF
import torchvision.transforms.functional as TF


def size_after_conv(h, ksize, stride=1, padding=0):
    return ((h - ksize + 2 * padding) // stride) + 1


def size_after_pooling(h, ksize):
    return h // ksize


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
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


def torch_log(x: th.Tensor) -> th.Tensor:
    return th.log(th.clamp(x, min=1e-10))


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: th.Tensor):
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

    def forward(self, x: th.Tensor, y: th.Tensor, as_loss: bool = True) -> th.Tensor:
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        # Compute means
        ux = NF.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        uy = NF.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)

        # Compute variances
        uxx = NF.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        uyy = NF.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        uxy = NF.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channel)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, channel: int, kernel_size: int, sigma: float) -> th.Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = th.arange(start, end, step=1, dtype=th.float)
        kernel_1d = th.exp(-th.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = th.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(channel, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d


class RL:
    def __init__(self):
        self.info = {}
        self.device = "cpu"

    def batch_to_tensor(self, batch, key_list=["states", "actions", "next_states", "rewards", "dones"]):
        return_list = {}
        for key in key_list:
            if isinstance(batch[key], th.Tensor):
                item = batch[key]
                if item.dtype != th.float:
                    item = item.to(th.float)
                if item.device != self.device:
                    item = item.to(self.device)
            else:
                item = th.tensor(batch[key], dtype=th.float, device=self.device)
            return_list[key] = item
        return return_list

    def update_from_batch(self, batch):
        raise NotImplementedError

    def get_action(self, state, deterministic=False):
        if isinstance(state, th.Tensor):
            state_tensor = state
            if state_tensor.dtype != th.float:
                state_tensor = state_tensor.to(th.float)
            if state_tensor.device != self.device:
                state_tensor = state_tensor.to(self.device)
        else:
            state_tensor = th.tensor(state, dtype=th.float, device=self.device)
        return state_tensor

    def save(self, path):
        th.save(self.state_dict(), path)

    def load(self, path):
        state_dict = th.load(path, map_location=self.device)
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
        return t.to("cpu").detach().numpy().copy()

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self._get_name() + "()"


class FE(nn.Module):
    optim: th.optim.Optimizer

    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor, return_pred: bool = False) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        raise NotImplementedError()

    def loss(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def __repr__(self):
        return self._get_name() + "()"


@dataclasses.dataclass
class Transition:
    state: np.ndarray
    next_state: np.ndarray
    reward: float
    action: np.ndarray
    terminated: dataclasses.InitVar[bool]
    truncated: dataclasses.InitVar[bool]
    success: bool = dataclasses.field(init=False)
    done: int = dataclasses.field(init=False)

    def __post_init__(self, terminated, truncated):
        self.success = terminated
        self.done = int(terminated or truncated)


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


class CropTrans(nn.Module):
    """
    画像の左右、または上下の両端を切り落として指定の大きさの正方形にリサイズする
    """

    def __init__(self, img_size=64):
        super().__init__()
        self.size = (img_size, img_size)

    def __call__(self, img: np.ndarray):
        s = min(img.shape[:2])
        _img = TF.center_crop(TF.to_tensor(img), (s, s))
        _img = cv2.resize(_img.detach().numpy().transpose((1, 2, 0)), self.size)
        return _img.transpose(2, 0, 1) * 0.5 + 0.5


class SegTree:
    def __init__(self, capacity, segfunc, init_value=0):
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.values = [init_value for _ in range(2 * capacity)]
        self.segfunc = segfunc

    def __str__(self):
        return str(self.values[self.capacity :])

    def __setitem__(self, idx, val):
        idx = idx + self.capacity
        self.values[idx] = val

        current_idx = idx // 2
        while current_idx >= 1:
            idx_lchild = 2 * current_idx
            idx_rchild = idx_lchild + 1
            self.values[current_idx] = self.segfunc(self.values[idx_lchild], self.values[idx_rchild])
            current_idx //= 2

    def __getitem__(self, idx):
        idx = idx + self.capacity
        return self.values[idx]

    def top(self):
        return self.values[1]


class SumTree(SegTree):
    def __init__(self, capacity, **kwargs):
        super().__init__(capacity, segfunc=lambda x, y: x + y, **kwargs)

    def sum(self):
        return self.top()

    def sample(self, z=None):
        z = random.uniform(0, self.sum()) if z is None else z
        assert 0 <= z <= self.sum(), f"z: {z}, self.sum(): {self.sum()}"

        current_idx = 1
        while current_idx < self.capacity:
            idx_lchild = 2 * current_idx
            idx_rchild = idx_lchild + 1
            if z > self.values[idx_lchild]:
                current_idx = idx_rchild
                z -= self.values[idx_lchild]
            else:
                current_idx = idx_lchild
        idx = current_idx - self.capacity
        return idx


class MinTree(SegTree):
    def __init__(self, capacity):
        super().__init__(capacity, segfunc=min, init_value=np.inf)

    def min(self):
        return self.top()
