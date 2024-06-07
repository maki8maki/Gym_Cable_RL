from typing import Callable, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import FE, SSIMLoss


class Flatten(nn.Module):
    def forward(self, input: th.Tensor):
        if input.ndim < 4:
            batch_size = 1
        else:
            batch_size = input.size(0)
        return input.view(batch_size, -1)


class UnFlatten(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, input: th.Tensor):
        if input.ndim < 4:
            batch_size = 1
        else:
            batch_size = input.size(0)
        return input.view(batch_size, self.size, 1, 1)


class ConvVAE(FE):
    def __init__(
        self,
        img_height,
        img_width,
        img_channel,
        hidden_dim,
        lr=1e-3,
        hidden_activation: Callable[[th.Tensor], th.Tensor] = F.tanh,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        with th.no_grad():
            n_flatten = self.encoder(th.zeros((1, img_channel, img_height, img_width), dtype=th.float)).numel()

        self.enc_mean = nn.Linear(n_flatten, hidden_dim)
        self.enc_var = nn.Linear(n_flatten, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, n_flatten),
            UnFlatten(size=n_flatten),
            nn.ConvTranspose2d(n_flatten, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channel, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.hidden_activation = hidden_activation
        self.optim = optim.Adam(self.parameters(), lr=lr)

        self.re_loss = SSIMLoss(channel=img_channel)

    def _encode(self, input: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        tmp = self.encoder(input)
        mu = self.enc_mean(tmp)
        log_var = self.enc_var(tmp)
        return mu, log_var

    def _decode(self, z: th.Tensor) -> th.Tensor:
        return self.decoder(z)

    def _reparameterize(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = th.randn_like(mu)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _bottleneck(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        return mu, log_var, z

    def forward(self, x: th.Tensor, return_pred: bool = False) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        _, _, z = self._bottleneck(x)
        if return_pred:
            y = self._decode(z)
            return z, y
        else:
            return self.hidden_activation(z)

    def loss(self, x: th.Tensor) -> th.Tensor:
        mu, log_var, z = self._bottleneck(x)
        kl = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp_())
        y = self._decode(z)
        re = self.re_loss(y, x) * x.numel()
        return kl + re
